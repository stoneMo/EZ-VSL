import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from model import EZVSL
from datasets import get_test_dataset, inverse_normalize
import cv2


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='ezvsl_vggss', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Models
    audio_visual_model = EZVSL(args.tau, args.out_dim)

    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            audio_visual_model.cuda(args.gpu)
            object_saliency_model.cuda(args.gpu)
            audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
            object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    testdataset = get_test_dataset(args)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("Loaded dataloader.")

    validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args)


@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.Evaluator()
    evaluator_obj = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    for step, (image, spec, bboxes, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        # Compute S_AVL
        heatmap_av = audio_visual_model(image.float(), spec.float())[0]
        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()

        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_av_obj = utils.normalize_img(pred_av * args.alpha + pred_obj * (1 - args.alpha))

            thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
            evaluator_av.cal_CIOU(pred_av, bboxes['gt_map'], thr_av)

            thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * 0.5)]
            evaluator_obj.cal_CIOU(pred_obj, bboxes['gt_map'], thr_obj)

            thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
            evaluator_av_obj.cal_CIOU(pred_av_obj, bboxes['gt_map'], thr_av_obj)

            if args.save_visualizations:
                denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                denorm_image = (denorm_image*255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_image.jpg'), denorm_image)

                # visualize bboxes on raw images
                gt_boxes_img = utils.visualize(denorm_image, bboxes['bboxes'])
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_gt_boxes.jpg'), gt_boxes_img)

                # visualize heatmaps
                heatmap_img = np.uint8(pred_av*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av.jpg'), fin)

                heatmap_img = np.uint8(pred_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_obj.jpg'), fin)

                heatmap_img = np.uint8(pred_av_obj*255)
                heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av_obj.jpg'), fin)

        print(f'{step+1}/{len(testdataloader)}: map_av={evaluator_av.finalize_AP50():.2f} map_obj={evaluator_obj.finalize_AP50():.2f} map_av_obj={evaluator_av_obj.finalize_AP50():.2f}')

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc

    print('AV: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av)))
    print('Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_obj)))
    print('AV_Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av_obj)))

    utils.save_iou(evaluator_av.ciou, 'av', args)
    utils.save_iou(evaluator_obj.ciou, 'obj', args)
    utils.save_iou(evaluator_av_obj.ciou, 'av_obj', args)


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


if __name__ == "__main__":
    main(get_arguments())

