import sys

sys.path.append('../')

import argparse
import os
import time
import shutil
import json
import datetime
import numpy as np
import warnings
import random

import torch
from torch import optim
import torch.backends.cuda as cudnn
from tensorboardX import SummaryWriter
from apex import amp

import my_optim
from utils import AverageMeter, MoveAverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore

from models import *

# default settings
ROOT_DIR = os.getcwd()
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'snapshot_bins')
IMG_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/images'))
train_list = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/list/train.txt'))
train_root_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/order_label.txt'))
train_parent_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/family_label.txt'))

test_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/test.txt'))

LR = 0.001
EPOCH = 21
DISP_INTERVAL = 20

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='DANet')
        self.parser.add_argument("--root_dir", type=str, default=ROOT_DIR,
                            help='Root dir for the project')
        self.parser.add_argument("--img_dir", type=str, default=IMG_DIR,
                            help='Directory of training images')
        self.parser.add_argument("--sim_alpha", type=float, default=0.05)
        self.parser.add_argument("--vis_name", type=str, default='')
        self.parser.add_argument("--train_list", type=str, default=train_list)
        self.parser.add_argument("--train_root_list", type=str, default=train_root_list)
        self.parser.add_argument("--train_parent_list", type=str, default=train_parent_list)
        self.parser.add_argument("--test_list", type=str, default=test_list)
        self.parser.add_argument("--batch_size", type=int, default=30)
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='cub')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_DA')
        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--diff_lr", type=str, default='True')
        self.parser.add_argument("--decay_points", type=str, default='80')
        self.parser.add_argument("--epoch", type=int, default=100)
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
        self.parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
        self.parser.add_argument("--log_dir", type=str, default='../log')
        self.parser.add_argument("--resume", type=str, default='False')
        self.parser.add_argument("--tencrop", type=str, default='False')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--eval_gcam", action='store_true', help='guided grad cam.')
        self.parser.add_argument("--mixp", action='store_true', help='turn on amp training.')
        self.parser.add_argument("--pretrained_model_dir", type=str, default='../pretrained_models')
        self.parser.add_argument("--pretrained_model", type=str, default='vgg16.pth')
        self.parser.add_argument("--seed", default=None, type=int, help='seed for initializing training. ')
        self.parser.add_argument("--mce", action='store_true', help='classification loss using cross entropy.')
        self.parser.add_argument("--bbce", action='store_true', help='classification loss using binary cross entropy loss with background.')
        self.parser.add_argument("--bce", action='store_true', help='classification loss using binary cross entropy loss.')
        self.parser.add_argument("--weight_bce", action='store_true', help='classification loss using weighted binary cross entropy loss.')
        self.parser.add_argument("--bce_pos_weight", type=float, default=0.9, help='the positive weight in weighted binary cross entropy loss.')
        self.parser.add_argument("--bbce_pos_weight", type=float, default=0.5, help='the positive weight in weighted binary cross entropy loss.')
        self.parser.add_argument("--sup", type=int, default=1)
        self.parser.add_argument("--lb", action='store_true',
                                 help='classification loss using both binary cross entropy loss and cross entropy loss.')
        self.parser.add_argument("--lb_bbce_weight", type=float, default=0.5, help='the bbce loss weight when training both bbce and ce.')
        self.parser.add_argument("--loss_trunc_th", type=float, default=1.0, help='The samples whose prediction score are higher than loss_trunc_th are detached.')
        self.parser.add_argument("--trunc_loss", action='store_true', help='switch on truncation loss.')
        self.parser.add_argument("--IN", action='store_true', help='switch on instance norm layer in first two conv blocks in Network.')
        self.parser.add_argument("--INL", action='store_true', help='switch on instance norm layer with learn affine parms in first two conv blocks in Network.')
        self.parser.add_argument("--RGAP", action='store_true', help='switch on residualized gap block.')
        self.parser.add_argument("--sc", action='store_true', help='switch on the class similar loss.')
        self.parser.add_argument("--sc_alpha", type=float, default=0.01, help='switch on the class similar loss.')
        self.parser.add_argument("--sc_old", type=float, default=0.5, help='the weight of old cls protype.')
        self.parser.add_argument("--sc_new", type=float, default=0.5, help='the weight of new cls protype.')
        self.parser.add_argument("--cls_th", type=float, default=0.1, help='the class threshold.')
        self.parser.add_argument("--cls_th_h", type=float, default=0.5, help='the class threshold.')
        self.parser.add_argument("--cls_th_l", type=float, default=0.1, help='the class threshold.')
        self.parser.add_argument("--bin_cls", action='store_true', help='switch on the binary classification.')
        self.parser.add_argument("--carafe", action='store_true', help='switch on the carafe.')
        self.parser.add_argument("--carafe_cls", action='store_true', help='switch on the carafe.')
        self.parser.add_argument("--non_local", action='store_true', help='switch on the non-local.')
        self.parser.add_argument("--nl_blocks", type=str, default='3,4,5', help='3 for feat3, etc.')
        self.parser.add_argument("--nl_residual", action='store_true', help='switch on the non-local with residual path.')
        self.parser.add_argument("--nl_kernel", type=int, default=-1, help='the kernel for non local module.')
        self.parser.add_argument("--nl_pairfunc", type=int, default=0, help='0 for guassian embedding, 1 for dot production')
        self.parser.add_argument("--avg_size", type=int, default=2, help='kernel size for average pooling.')
        self.parser.add_argument("--avg_stride", type=int, default=2, help='stride size for average pooling.')
        self.parser.add_argument("--avg_bin", action='store_true', help='switch on average pooling for binary location map.')
        self.parser.add_argument("--adap_w", action='store_true', help='switch on adaptively weighting for binary location map.')
        self.parser.add_argument("--adap_w_gama", type=float, default=2, help='epison ratio for adaptive weight for binary location.')
        self.parser.add_argument("--bak_fac", type=float, default=0.3, help='the class threshold.')
        self.parser.add_argument("--sep_loss", action='store_true', help='switch on calculating loss for each individual.')
        self.parser.add_argument("--loc_branch", action='store_true', help='switch on location branch.')
        self.parser.add_argument("--com_feat", action='store_true', help='switch on location branch.')
        self.parser.add_argument("--th_bg", type=float, default=0.2, help='the variance threshold for back ground.')
        self.parser.add_argument("--th_fg", type=float, default=0.5, help='the class threshold for fore ground.')
        self.parser.add_argument("--loc_start", type=float, default=10, help='the start epoch to add location loss.')
        self.parser.add_argument("--cls_start", type=float, default=120, help='the start epoch to modify classification using location prediction.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str = opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        return opt

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def get_model(args):
    amp.register_float_function(torch, 'sigmoid')
    model = eval(args.arch).model(pretrained=True,
                                  num_classes=args.num_classes,
                                  args=args)
    model.to(args.device)


    lr = args.lr
    added_layers = ['cls', 'classifier'] if args.diff_lr == 'True' else []
    weight_list = []
    bias_list = []
    added_weight_list = []
    added_bias_list = []
    print('\n following parameters will be assigned 10x learning rate:')
    for name, value in model.named_parameters():
        if any([x in name for x in added_layers]):
            print(name)
            if 'weight' in name:
                added_weight_list.append(value)
            elif 'bias' in name:
                added_bias_list.append(value)
        else:
            if 'weight' in name:
                weight_list.append(value)
            elif 'bias' in name:
                bias_list.append(value)

    optimizer = optim.SGD([{'params': weight_list, 'lr': lr},
                           {'params': bias_list, 'lr': lr * 2},
                           {'params': added_weight_list, 'lr': lr * 10},
                           {'params': added_bias_list, 'lr': lr * 20}],
                          momentum=0.9, weight_decay=0.0005, nesterov=True)
    if args.mixp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model = torch.nn.DataParallel(model, args.gpus)
    if args.resume == 'True':
        restore(args, model, optimizer, including_opt=False)
    return model, optimizer


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    # for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('#epoch \t loss \t pred@1 \t pred@5\n')

    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_loc = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    protype_v = protype_h = None
    if args.sc:
        protype_h = MoveAverageMeter(args.num_classes, 7, old=args.sc_old, new=args.sc_new)
        protype_v = MoveAverageMeter(args.num_classes, 7, old=args.sc_old, new=args.sc_new)

    args.device = torch.device('cuda') if args.gpus[0] >= 0 else torch.device('cpu')
    model, optimizer = get_model(args)

    model.train()
    train_loader, _, _ = data_loader(args)



    # construct writer
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch
    end = time.time()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)
    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        losses_loc.reset()
        top1.reset()
        top5.reset()

        batch_time.reset()
        res = my_optim.reduce_lr(args, optimizer, current_epoch)

        if res:
            with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
                for g in optimizer.param_groups:
                    out_str = 'Epoch:%d, %f\n' % (current_epoch, g['lr'])
                    fw.write(out_str)

        steps_per_epoch = len(train_loader)
        for idx, dat in enumerate(train_loader):
            img_path, img, label = dat
            global_counter += 1
            img, root_label, parent_label, child_label = img.to(args.device), label[0].to(args.device), \
                                                         label[1].to(args.device), label[2].to(args.device)

            logits = model(img)
            if args.sc:
                child_map_h, child_map_v = model.module.child_cls_fea
                protype_h.update(child_map_h.detach(), child_label.long())
                protype_v.update(child_map_v.detach(), child_label.long())


            loss_val, loss_loc = model.module.get_loss(logits,child_label,
                    protype_h = protype_h, protype_v=protype_v, epoch=current_epoch,
                    loc_start=args.loc_start, cls_start=args.cls_start)

            # write into tensorboard
            writer.add_scalar('loss_val', loss_val, global_counter)

            # network parameter update
            optimizer.zero_grad()
            if args.mixp:
                with amp.scale_loss(loss_val, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_val.backward()
            optimizer.step()

            logits = torch.mean(torch.mean(logits, dim=2), dim=2)
            if not args.onehot == 'True':
                prec1, prec5 = evaluate.accuracy(logits.data, child_label.long(), topk=(1, 5))
                top1.update(prec1[0], img.size()[0])
                top5.update(prec5[0], img.size()[0])


            losses.update(loss_val.data, img.size()[0])
            losses_loc.update(loss_loc.data, img.size()[0])

            batch_time.update(time.time() - end)

            end = time.time()

            if global_counter % args.disp_interval == 0:
                # Calculate ETA
                eta_seconds = ((total_epoch - current_epoch) * steps_per_epoch +
                               (steps_per_epoch - idx)) * batch_time.avg
                eta_str = "{:0>8}".format(str(datetime.timedelta(seconds=int(eta_seconds))))
                eta_seconds_epoch = steps_per_epoch * batch_time.avg
                eta_str_epoch = "{:0>8}".format(str(datetime.timedelta(seconds=int(eta_seconds_epoch))))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ETA {eta_str}({eta_str_epoch})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_loc {loss_loc.val:.4f} ({loss_loc.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader), batch_time=batch_time,
                    eta_str=eta_str, eta_str_epoch=eta_str_epoch, loss=losses, loss_loc=losses_loc, top1=top1, top5=top5))
                writer.add_scalar('top1', top1.avg, global_counter)
                writer.add_scalar('top5', top5.avg, global_counter)


        current_epoch += 1
        if current_epoch % 50 == 0:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'arch': args.arch,
                                'global_counter': global_counter,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()
                            }, is_best=False,
                            filename='%s_epoch_%d_glo_step_%d.pth.tar'
                                     % (args.dataset, current_epoch, global_counter))

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            fw.write('%d \t %.4f \t %.4f \t %.3f \t %.3f\n' % (current_epoch, losses.avg, losses_loc.avg, top1.avg, top5.avg))

        losses.reset()
        losses_loc.reset()
        top1.reset()
        top5.reset()



if __name__ == '__main__':
    args = opts().parse()
    train(args)
