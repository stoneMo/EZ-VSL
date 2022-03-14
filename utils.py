import os
import json
from torch.optim import *
import numpy as np
from sklearn import metrics
import xml.etree.ElementTree as ET

class Evaluator():

    def __init__(self):
        super(Evaluator, self).__init__()
        self.ciou = []

    def cal_CIOU(self, infer, gtmap, thres=0.01):

        infer_map = np.zeros((224, 224))
        infer_map[infer>=thres] = 1

        ciou = np.sum(infer_map*gtmap) / (np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))

        self.ciou.append(ciou)
        return ciou, np.sum(infer_map*gtmap),(np.sum(gtmap)+np.sum(infer_map*(gtmap==0)))


    def cal_AUC(self):
        results = []
        for i in range(21):
            result = np.sum(np.array(self.ciou)>=0.05*i)
            result = result / len(self.ciou)
            results.append(result)
        x = [0.05*i for i in range(21)]
        auc = metrics.auc(x, results)
        print(results)
        return auc

    def final(self):
        ciou = np.mean(np.array(self.ciou)>=0.5)
        return ciou

    def clear(self):
        self.ciou = []


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def testset_gt(args,name):

    if args.testset == 'flickr':
        gt = ET.parse(args.test_gt_path + '%s.xml' % name).getroot()
        gt_map = np.zeros([224, 224])
        bboxs = []
        for child in gt: 
            for childs in child:
                bbox = []
                if childs.tag == 'bbox':
                    for index,ch in enumerate(childs):
                        if index == 0:
                            continue
                        bbox.append(int(224 * int(ch.text)/256))
                bboxs.append(bbox)
        for item_ in bboxs:
            temp = np.zeros([224,224])
            (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
            gt_map += temp
        gt_map /= 2
        gt_map[gt_map>1] = 1

        bboxs_list = bboxs
        
    elif args.testset.lower() in {'vggss', 'vggss_heard', 'vggss_unheard'}:
        gt = args.gt_all[name]
        gt_map = np.zeros([224,224])
        bboxs = []
        for item_ in gt:
            item_ =  list(map(lambda x: int(224* max(x,0)), item_) )
            temp = np.zeros([224,224])
            (xmin, ymin, xmax, ymax) = item_[0],item_[1],item_[2],item_[3]
            temp[ymin:ymax,xmin:xmax] = 1
            gt_map += temp
            bboxs.append([xmin,ymin,xmax,ymax])
        gt_map[gt_map>0] = 1

        bboxs_list = bboxs

    return gt_map, bboxs_list


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

def save_iou(iou_list, suffix, args):
    # sorted iou
    sorted_iou = np.sort(iou_list).tolist()
    sorted_iou_indices = np.argsort(iou_list).tolist()
    file_iou = open(os.path.join(args.output_path,"iou_test_{}.txt".format(suffix)),"w")
    for indice, value in zip(sorted_iou_indices, sorted_iou):
        line = str(indice) + ',' + str(value) + '\n'
        file_iou.write(line)
    file_iou.close()