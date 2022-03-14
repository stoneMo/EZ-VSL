import torch
from torch import nn
import torch.nn.functional as F
from models import base_models
from torchvision.models import resnet18, resnet50


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor


def normalize_img(value, vmax=None, vmin=None):
    #  pdb.set_trace()
    value1 = value.view(value.size(0), -1)
    value1 -= value1.min(1, keepdim=True)[0]
    value1 /= value1.max(1, keepdim=True)[0]
    return value1.view(value.size(0), value.size(1), value.size(2), value.size(3))


class VSLNet(nn.Module):
    def __init__(self, args):
        super(VSLNet, self).__init__()

        # -----------------------------------------------
        self.imgnet = resnet18(pretrained=True)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()
        self.img_proj = nn.Conv2d(512, args.out_dim, kernel_size=(1, 1))
        if args.dilated:
            for it, block in enumerate(self.imgnet.layer4):
                save_weight = block.conv1.weight
                if it == 0:
                    block.conv1 = nn.Conv2d(save_weight.shape[1], save_weight.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                else:
                    block.conv1 = nn.Conv2d(save_weight.shape[1], save_weight.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
                block.conv1.weight.data.copy_(save_weight)

                if block.downsample is not None:
                    assert it == 0
                    save_weight = block.downsample[0].weight
                    block.downsample[0] = nn.Conv2d(save_weight.shape[1], save_weight.shape[0], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
                    block.downsample[0].weight.data.copy_(save_weight)

                save_weight = block.conv2.weight
                block.conv2 = nn.Conv2d(save_weight.shape[1], save_weight.shape[0], kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
                block.conv2.weight.data.copy_(save_weight)

        self.audnet = resnet18()
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()
        self.aud_proj = nn.Linear(512, args.out_dim)

        self.tau = args.tau
        self.args = args

        for net in [self.audnet, self.img_proj, self.aud_proj]:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.normal_(m.weight, mean=1, std=0.02)
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, audio):
        # Image
        img_f = self.imgnet(image)
        img_f = img_f.unflatten(1, (512, 7, 7)) if not self.args.dilated else img_f.unflatten(1, (512, 14, 14))
        img = self.img_proj(img_f)    # [bs, 512, 14, 14]

        # Audio
        aud = self.audnet(audio)
        aud = self.aud_proj(aud)    # [bs, 512]

        # Join them
        img = nn.functional.normalize(img, dim=1)
        aud = nn.functional.normalize(aud, dim=1)
        A = torch.einsum('nchw,nc->nhw', img, aud).unsqueeze(1) / self.tau
        A0 = torch.einsum('nchw,mc->nmhw', img, aud) / self.tau
        return A, A0, img_f