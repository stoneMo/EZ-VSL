import os
import json
from torch.optim import *
import numpy as np
from sklearn import metrics


class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):
        infer_map = np.zeros((224, 224))
        infer_map[infer >= thres] = 1
        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap==0)))

        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap), (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

    def finalize_AUC(self):
        cious = [np.sum(np.array(self.ciou) >= 0.05*i) / len(self.ciou)
                 for i in range(21)]
        thr = [0.05*i for i in range(21)]
        auc = metrics.auc(thr, cious)
        return auc

    def finalize_AP50(self):
        ap50 = np.mean(np.array(self.ciou) >= 0.5)
        return ap50

    def finalize_cIoU(self):
        ciou = np.mean(np.array(self.ciou))
        return ciou

    def clear(self):
        self.ciou = []


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def visualize(raw_image, boxes):
    import cv2
    boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]

    for box in boxes:

        xmin,ymin,xmax,ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])

        cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    return boxes_img[:, :, ::-1]


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def save_iou(iou_list, suffix, output_dir):
    # sorted iou
    sorted_iou = np.sort(iou_list).tolist()
    sorted_iou_indices = np.argsort(iou_list).tolist()
    file_iou = open(os.path.join(output_dir,"iou_test_{}.txt".format(suffix)),"w")
    for indice, value in zip(sorted_iou_indices, sorted_iou):
        line = str(indice) + ',' + str(value) + '\n'
        file_iou.write(line)
    file_iou.close()
