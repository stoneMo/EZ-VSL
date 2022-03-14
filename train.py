from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils import *
from losses import *
import numpy as np
import json
import argparse
from model import VSLNet
from datasets.dataloader import GetAudioVideoDataset_Train, GetAudioVideoDataset_Test

from torch import multiprocessing as mp
import random
import os
import builtins
import torch.distributed as dist
import time


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset',default='vggss',type=str,help='testset,(flickr or vggss)') 
    parser.add_argument('--testset',default='vggss',type=str,help='testset,(flickr or vggss)') 
    parser.add_argument('--train_data_path', default='',type=str,help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='',type=str,help='Root directory path of test data')
    parser.add_argument('--test_gt_path',default='',type=str)
    parser.add_argument('--image_size',default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--dilated', action='store_true')
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # training/evaluation parameters
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument('--freeze_vision', action='store_true')
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
    parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
    parser.add_argument("--eval_period", type=int, default=100, help="training loss and eval print period")
    parser.add_argument('--pretrained_dir', type=str, default=None, help='path to pretrained model weights')

    parser.add_argument('--model_dir', type=str, default='checkpoints', help='path to save trained model weights')
    parser.add_argument('--model_name', type=str, default='vslnet', help='model type (vslnet)')
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


def main():
    mp.set_start_method('spawn')

    args = get_arguments()
    if args.gpu == -1:
        args.gpu = None
    args.port = random.randint(10000, 20000)
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model dir
    home_dir = os.path.join(args.model_dir, args.trainset, '_'.join([args.model_name, "bs"+str(args.batch_size),
                                                        "lr"+str(args.init_lr)]))
    if args.freeze_vision:
        home_dir += '_freezevision'
    if args.out_dim != 512:
        home_dir += f'_out{args.out_dim}'
    if args.dilated:
        home_dir += f'_dilated'
    
    if args.suffix is not None:
        home_dir = home_dir + '_' + args.suffix
    model_dir = os.path.join(home_dir, "model")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # load model
    if args.model_name == 'vslnet':
        model = VSLNet(args)
    else:
        raise ValueError

    if args.freeze_vision:
        for m in model.imgnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                continue
            for p in m.parameters():
                p.requires_grad = False

    print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    if args.pretrained_dir:
        checkpoint = torch.load(args.pretrained_dir)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if torch.cuda.is_available():
            model.cuda()
        print('load pretrained model.')

    # dataloader
    traindataset = GetAudioVideoDataset_Train(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    testdataset = GetAudioVideoDataset_Test(args)
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    # gt for vggss
    if args.testset == 'vggss':
        args.gt_all = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            args.gt_all[annotation['file']] = annotation['bbox']

    args.num_train_steps = len(train_loader) * args.epochs

    # optimizer/scheduler
    # optimizer, scheduler = build_optimizer_and_scheduler(model, args)      # use sgd
    optimizer, scheduler = build_optimizer_and_scheduler_adam(model, args)

    log_writer = open(os.path.join(model_dir, "log.txt"), mode="w", encoding="utf-8")
    # tb_writer = SummaryWriter(logdir=os.path.join(home_dir, "summary_dir"))

    start_epoch = 0
    best_cIoU, best_Auc = 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu')
        start_epoch = ckp['epoch']
        best_cIoU = ckp['best_cIoU']
        best_Auc = ckp['best_Auc']
        if torch.cuda.is_available():
            model.load_state_dict(ckp['model'])
        else:
            model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    cIoU, auc = validate(test_loader, model, start_epoch, log_writer, None, args)
    print(f'cIoU (epoch {start_epoch}): {cIoU}')
    print(f'AUC (epoch {start_epoch}): {auc}')
    print(f'best_cIoU: {best_cIoU}')
    print(f'best_Auc: {best_Auc}')

    for epoch in range(start_epoch, args.epochs):
        print(model_dir)
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, epoch, log_writer, None, args)

        cIoU, auc = validate(test_loader, model, epoch, log_writer, None, args)
        if args.rank == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'best_cIoU': best_cIoU, 'best_Auc': best_Auc}, os.path.join(model_dir, 'latest.pth'))
        if cIoU >= best_cIoU:
            best_cIoU = cIoU
            best_Auc = auc
            if args.rank == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, 'best_cIoU': best_cIoU, 'best_Auc': best_Auc}, os.path.join(model_dir, 'best.pth'))

        print(f'cIoU (epoch {epoch+1}): {cIoU}')
        print(f'AUC (epoch {epoch+1}): {auc}')
        print(f'best_cIoU: {best_cIoU}')
        print(f'best_Auc: {best_Auc}')


def train(train_loader, model, optimizer, epoch, log_writer, tb_writer, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr],
        prefix="Epoch: [{}]".format(epoch),
        fp=log_writer
    )

    end = time.time()
    for i, (image, spec, audio, name) in enumerate(train_loader):
        data_time.update(time.time() - end)

        global_step = i + len(train_loader) * epoch

        # print('%d / %d' % (i,len(train_loader) - 1))
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        # print("image:", image.shape, type(image))    # [bs, 3, 224, 224]
        # print("im:", im.shape)                       # [bs, 256, 256, 3]
        # print("spec:", spec.shape, type(spec))       # [bs, 1, 257, 925]

        _, S, embed_img = model(image.float(), spec.float())

        # print("logits:", logits.shape)        # [bs, 1+bs+1]
       
        loss = ce_loss(S)
        loss_mtr.update(loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        torch.cuda.empty_cache()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            if args.rank == 0:
                progress.display(i)
        del loss


def validate(test_loader, model, epoch, log_writer, tb_writer, args):
    model.train(False)
    iou = []
    for step, (image, spec, audio, name) in enumerate(test_loader):
        # print('%d / %d' % (step,len(test_loader) - 1))
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        heatmap = model(image.float(), spec.float())[0]
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_arr = heatmap.data.cpu().numpy()

        for i in range(spec.shape[0]):
            gt_map, boxes = testset_gt(args, name[i])
            pred = normalize_img(heatmap_arr[i, 0])
            threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            evaluator = Evaluator()
            ciou, inter, union = evaluator.cal_CIOU(pred,gt_map,threshold)
            iou.append(ciou)

    results = []
    for i in range(21):
        result = np.sum(np.array(iou) >= 0.05 * i)
        result = result / len(iou)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc = metrics.auc(x, results)
    cIoU = np.sum(np.array(iou) >= 0.5)/len(iou)
    # print('cIoU',cIoU)
    # print('auc',auc)

    return cIoU, auc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()

