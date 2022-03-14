import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import json
import argparse
import csv
from model import VSLNet
from datasets import GetAudioVideoDataset_Test, inverse_normalize
import cv2
from sklearn import metrics
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', default='flickr',type=str,help='testset,(flickr or vggss)')
    parser.add_argument('--testset', default='flickr',type=str,help='testset,(flickr or vggss)')
    parser.add_argument('--test_data_path', default='',type=str,help='Root directory path of data')
    parser.add_argument('--image_size', default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--test_gt_path', default='',type=str)
    parser.add_argument('--output_path', default='./visualization',type=str)
    parser.add_argument('--summaries_dir', default='',type=str,help='Model path')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument('--use_latest',action='store_true')
    parser.add_argument('--out_dim', default=512, type=int)

    parser.add_argument('--iou_thres', default=0.5, type=float, help='iou threshold')
    parser.add_argument('--alpha', default=0.9, type=float, help='tau')
    parser.add_argument('--pred_fraction', default=0.5, type=float, help='tau')
    parser.add_argument('--freeze_vision', action='store_true')

    parser.add_argument('--model_dir', type=str, default='checkpoints', help='path to save trained model weights')
    parser.add_argument('--model_name', type=str, default='svlnet-v2', help='model type (avenet, svlnet)')
    parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')

    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class MaxReducer(nn.Module):
    def __init__(self, dim):
        super(MaxReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(self.dim)[0]


class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


def main():
    args = get_arguments()
    if args.gpu == -1:
        args.gpu = None

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # create model dir    
    home_dir = os.path.join(args.model_dir, args.trainset, '_'.join([args.model_name, "bs"+str(args.batch_size),
                                                          "lr"+str(args.init_lr)]))
    if args.freeze_vision:
        home_dir += '_freezevision'
    if args.out_dim != 512:
        home_dir += f'_out{args.out_dim}'
    if 'dilated' in args and args.dilated:
        home_dir += f'_dilated'

    if args.suffix is not None:
        home_dir = home_dir + '_' + args.suffix
    model_dir = os.path.join(home_dir, "model")

    # load model
    if args.model_name == 'vslnet':
        model = VSLNet(args)
    else:
        raise ValueError

    from torchvision.models import resnet18
    vision_model = resnet18(pretrained=True)
    vision_model.avgpool = nn.Identity()
    vision_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    ckp_fn = 'latest.pth' if args.use_latest else 'best.pth'
    if os.path.exists(os.path.join(model_dir, ckp_fn)):
        ckp = torch.load(os.path.join(model_dir, ckp_fn), map_location='cpu')
        try:
            model.load_state_dict(ckp['model'])
        except Exception:
            model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {os.path.join(model_dir, ckp_fn)}")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            vision_model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            vision_model = torch.nn.parallel.DistributedDataParallel(vision_model, device_ids=[args.gpu])

    # dataloader
    testdataset = GetAudioVideoDataset_Test(args)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("Loaded dataloader.")

    validate(testdataloader, model, vision_model, args)


@torch.no_grad()
def validate(testdataloader, model, vision_model, args):
    # gt for vggss
    if args.testset in {'vggss', 'vggss_heard', 'vggss_unheard'}:
        args.gt_all = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            args.gt_all[annotation['file']] = annotation['bbox']

    model.train(False)
    vision_model.train(False)
    evaluator = Evaluator()
    iou_av, iou_obj, iou_av_obj = [], [], []
    global_step = 0
    for step, (image, spec, audio, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        heatmap_av = model(image.float(), spec.float())[0]
        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()

        img_feat = vision_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()
        for i in range(spec.shape[0]):
            pred_av = normalize_img(heatmap_av[i, 0])
            pred_obj = normalize_img(heatmap_obj[i, 0])
            pred_av_obj = normalize_img(pred_av * args.alpha + pred_obj * (1 - args.alpha))
            # print("heatmap_now:", heatmap_av_now.shape, type(heatmap_av_now))  # (224, 224)

            denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            denorm_image = (denorm_image*255).astype(np.uint8)
            # print("denorm_image:", denorm_image.shape) # [1, 3, 224, 224]
            cv2.imwrite(os.path.join(args.output_path,'image_'+str(global_step)+'.jpg'), denorm_image)

            # merge pred_av maps and frame
            heatmap_img = np.uint8(pred_av*255)
            heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
            fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
            cv2.imwrite(os.path.join(args.output_path, 'pred_av_'+str(global_step)+'.jpg'), fin)
            
            # merge pred_obj maps and frame
            heatmap_img = np.uint8(pred_obj*255)
            heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
            fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
            cv2.imwrite(os.path.join(args.output_path, 'pred_obj_'+str(global_step)+'.jpg'), fin)

            # merge pred_av_obj maps and frame
            heatmap_img = np.uint8(pred_av_obj*255)
            heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
            fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
            cv2.imwrite(os.path.join(args.output_path, 'pred_av_obj_'+str(global_step)+'.jpg'), fin)


            gt_map, boxes = testset_gt(args, name[i])
                        
            # visualize gt map
            # gt_map_image = np.uint8(gt_map*120)
            # gt_map_image = cv2.applyColorMap(gt_map_image[:, :, np.newaxis], cv2.COLORMAP_JET)
            # gt_map_image = gt_map_image[:, :, np.newaxis]
            # gt_map_image = cv2.cvtColor(gt_map[:, :, np.newaxis],cv2.COLOR_GRAY2RGB)
            # fin_gt = cv2.addWeighted(gt_map_image, 0.3, np.uint8(denorm_image), 0.7, 0)
            # cv2.imwrite(os.path.join(args.output_path, 'gt_heatmap_'+str(global_step)+'.jpg'), gt_map_image)

            # visualize bboxes on raw images
            gt_boxes_img = visualize(denorm_image, boxes)
            cv2.imwrite(os.path.join(args.output_path,'gt_boxes_'+str(global_step)+'.jpg'), gt_boxes_img)

            threshold_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * args.pred_fraction)]
            ciou, inter, union = evaluator.cal_CIOU(pred_av, gt_map, threshold_av)
            iou_av.append(ciou)

            threshold_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * args.pred_fraction)]
            ciou, inter, union = evaluator.cal_CIOU(pred_obj, gt_map, threshold_obj)
            iou_obj.append(ciou)

            threshold_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * args.pred_fraction)]
            ciou, inter, union = evaluator.cal_CIOU(pred_av_obj, gt_map, threshold_av_obj)
            iou_av_obj.append(ciou)

            global_step += 1

        print(f'{step+1}/{len(testdataloader)}: map_av={np.sum(np.array(iou_av) >= args.iou_thres)/len(iou_av)*100:.2f} map_obj={np.sum(np.array(iou_obj) >= args.iou_thres)/len(iou_obj)*100:.2f} map_av_obj={np.sum(np.array(iou_av_obj) >= args.iou_thres)/len(iou_av_obj)*100:.2f}')

    def compute_stats(iou_list):
        results = []
        for i in range(21):
            result = np.sum(np.array(iou_list) >= 0.05 * i)
            result = result / len(iou_list)
            results.append(result)
        x = [0.05 * i for i in range(21)]
        mAP = np.sum(np.array(iou_list) >= args.iou_thres)/len(iou_list)
        ciou = np.array(iou_list).mean()
        auc = metrics.auc(x, results)
        return mAP, ciou, auc

    print('AV (AP50, Avg-cIoU, AUC)', compute_stats(iou_av))
    print('Obj (AP50, Avg-cIoU, AUC)', compute_stats(iou_obj))
    print('Av+Obj (AP50, Avg-cIoU, AUC)', compute_stats(iou_av_obj))

    save_iou(iou_av, 'av', args)
    save_iou(iou_obj, 'obj', args)
    save_iou(iou_av_obj, 'av_obj', args)


if __name__ == "__main__":
    main()

