import sys

sys.path.append('../')
import argparse
import os
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn.functional as F

from utils import AverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore
from utils.localization import get_topk_boxes, get_topk_boxes_hier
from utils.vistools import save_im_heatmap_box, save_im_gcam_ggrads
from models import *

# default settings

LR = 0.001
EPOCH = 200
DISP_INTERVAL = 50

# default settings
ROOT_DIR = os.getcwd()
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'snapshot_bins')
IMG_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/images'))
train_list = os.path.abspath(os.path.join(ROOT_DIR, '../data', 'CUB_200_2011/list/train.txt'))
train_root_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/order_label.txt'))
train_parent_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/family_label.txt'))

test_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/test.txt'))
testbox_list = os.path.abspath(os.path.join(ROOT_DIR, '../data','CUB_200_2011/list/test_boxes.txt'))
class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='ECCV')
        self.parser.add_argument("--root_dir", type=str, default='')
        self.parser.add_argument("--img_dir", type=str, default=IMG_DIR)
        self.parser.add_argument("--train_list", type=str, default=train_list)
        self.parser.add_argument("--train_root_list", type=str, default=train_root_list)
        self.parser.add_argument("--train_parent_list", type=str, default=train_parent_list)
        self.parser.add_argument("--cos_alpha", type=float, default=0.2)
        self.parser.add_argument("--num_maps", type=float, default=8)
        self.parser.add_argument("--test_list", type=str, default=test_list)
        self.parser.add_argument("--test_box", type=str, default=testbox_list)
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--input_size", type=int, default=256)
        self.parser.add_argument("--crop_size", type=int, default=224)
        self.parser.add_argument("--dataset", type=str, default='imagenet')
        self.parser.add_argument("--num_classes", type=int, default=200)
        self.parser.add_argument("--arch", type=str, default='vgg_v0')
        self.parser.add_argument("--threshold", type=str, default='0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45')
        self.parser.add_argument("--lr", type=float, default=LR)
        self.parser.add_argument("--decay_points", type=str, default='none')
        self.parser.add_argument("--epoch", type=int, default=EPOCH)
        self.parser.add_argument("--tencrop", type=str, default='True')
        self.parser.add_argument("--onehot", type=str, default='False')
        self.parser.add_argument("--gpus", type=str, default='0', help='-1 for cpu, split gpu id by comma')
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
        self.parser.add_argument("--snapshot_dir", type=str, default='')
        self.parser.add_argument("--resume", type=str, default='True')
        self.parser.add_argument("--restore_from", type=str, default='')
        self.parser.add_argument("--global_counter", type=int, default=0)
        self.parser.add_argument("--current_epoch", type=int, default=0)
        self.parser.add_argument("--debug", action='store_true', help='.')
        self.parser.add_argument("--debug_dir", type=str, default='../debug', help='save visualization results.')
        self.parser.add_argument("--eval_gcam", action='store_true', help='.')
        self.parser.add_argument("--eval_g2", action='store_true', help='.')
        self.parser.add_argument("--mce", action='store_true',
                                 help='classification loss using cross entropy.')
        self.parser.add_argument("--bbce", action='store_true',
                                 help='classification loss using binary cross entropy loss with background.')
        self.parser.add_argument("--bce", action='store_true',
                                 help='classification loss using binary cross entropy loss.')
        self.parser.add_argument("--sup", type=int, default=1)
        self.parser.add_argument("--lb", action='store_true',
                                 help='classification loss using both binary cross entropy loss and cross entropy loss.')
        self.parser.add_argument("--NoHDA", action='store_true',help='switch off the hda.')
        self.parser.add_argument("--NoDDA", action='store_true',help='switch off the dda.')
        self.parser.add_argument("--bg", action='store_true',help='grad_cam for background.')
        self.parser.add_argument("--IN", action='store_true', help='switch off instance norm layer in first two conv blocks in Network.')
        self.parser.add_argument("--INL", action='store_true', help='switch off instance norm layer with learn affine parms in first two conv blocks in Network.')
        self.parser.add_argument("--loss_trunc_th", type=float, default=1.0, help='The samples whose prediction score are higher than loss_trunc_th are detached.')
        self.parser.add_argument("--trunc_loss", action='store_true', help='switch on truncation loss.')
        self.parser.add_argument("--RGAP", action='store_true', help='switch on residualized gap block.')
        self.parser.add_argument("--sc", action='store_true', help='switch on the class similar loss.')
        self.parser.add_argument("--bin_cls", action='store_true', help='switch on the binary classification.')
        self.parser.add_argument("--carafe", action='store_true', help='switch on the carafe.')
        self.parser.add_argument("--carafe_cls", action='store_true', help='switch on the carafe.')
        self.parser.add_argument("--non_local", action='store_true', help='switch on the non-local.')
        self.parser.add_argument("--non_local_res", action='store_true', help='switch on the non-local with residual path.')
        self.parser.add_argument("--non_local_kernel", type=int, default=3, help='the kernel for non local module.')
        self.parser.add_argument("--non_local_pf", type=int, default=0, help='0 for guassian embedding, 1 for dot production')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus_str=opt.gpus
        opt.gpus = list(map(int, opt.gpus.split(',')))
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0]>=0 else [-1]
        opt.threshold = list(map(float, opt.threshold.split(',')))
        opt.num_maps = int(opt.num_maps)
        if 'vgg' in opt.arch:
            opt.size=(28, 28)
        else:
            opt.size=(25, 25)
        return opt

def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes, args=args)

    model = torch.nn.DataParallel(model, args.gpus)
    model.cuda()

    if args.resume == 'True':
        restore(args, model, None, istrain=False)

    return model

def get_grad(model, logits, feat, num_maps, layer='cls3', topk=(1,), logits_ce=None, bg=False):
    grads = {}
    gcam = None
    g2 = None
    maxk = max(topk)
    if logits_ce is not None:
        pred_clses = torch.argsort(logits_ce.view(-1), descending=True)[:maxk]
    else:
        if bg:
            pred_clses = torch.argsort(logits.view(-1)[:-1], descending=True)[:maxk]
            # pred_clses = (-1,)*maxk
        else:
            pred_clses = torch.argsort(logits.view(-1), descending=True)[:maxk]
    feat_clone = feat.clone()
    n, _, h, w = feat_clone.size()
    feat_clone[feat_clone > 0] = 1
    feat_clone = feat_clone.view(n, -1, num_maps, h, w)
    for i, pred_cls in enumerate(pred_clses):
        grad_traget_map = torch.zeros_like(logits)
        grad_traget_map[0,pred_cls] = 1
        if bg:
            grad_traget_map[0, -1] = -1
        model.zero_grad()
        logits.backward(grad_traget_map, retain_graph=True)
        if g2 is None:
            g2 = model.module.guided_grad.max(dim=1, keepdim=True)[0]
        else:
            g2 = torch.cat((g2,model.module.guided_grad.max(dim=1, keepdim=True)[0]), dim=1)

        last_layer_grad = model.module.last_layer_grad_out[layer].view(n, -1, num_maps, h, w)
        last_layer_grad = last_layer_grad * feat_clone
        last_layer_grad_w = last_layer_grad.mean(dim=3,keepdim=True).mean(dim=4, keepdim=True)
        feat = feat.view(n, -1, num_maps, h, w)
        gcam_i = (feat*last_layer_grad_w).sum(dim=2)/(last_layer_grad_w.sum(dim=2)+1e-13)
        # gcam_i = feat.mean(dim=2)
        gcam_i = gcam_i[:,int(pred_cls):int(pred_cls)+1,:,:]
        # gcam_i = gcam_i[:,int(pred_cls):,:,:]
        if layer != 'cls3':
            gcam_i = F.interpolate(gcam_i, size=args.size, mode='bilinear', align_corners=True)
        if gcam is None:
            gcam = gcam_i
        else:
            gcam = torch.cat((gcam,gcam_i),dim=1)
    grads['gcam_{}'.format(layer)] = gcam
    grads['g2_{}'.format(layer)] = g2
    return grads

def eval_loc(logit_child, logit_parent, logit_root, child_maps,parent_maps,root_maps, img_path, input_size, crop_size, label, gt_boxes,
             topk=(1,5), threshold=None, mode='union', debug=False, debug_dir=None, gcam=False, g2=False, NoHDA=False, bin_map=None):
    top_boxes, top_maps = get_topk_boxes_hier(logit_child[0], logit_parent[0], logit_root[0], child_maps,
                                              parent_maps, root_maps, img_path, input_size,
                                              crop_size, topk=topk, threshold=threshold, mode=mode, gcam=gcam, g2=g2,
                                              NoHDA=NoHDA, bin_map=bin_map)
    top1_box, top5_boxes = top_boxes

    # update result record
    deterr_1,locerr_1, deterr_5, locerr_5 = evaluate.locerr((top1_box, top5_boxes), label.data.long().numpy(), gt_boxes,
                                         topk=(1, 5))

    # if debug and threshold==0.15:
    #     save_im_heatmap_box(img_path, top_maps, top5_boxes, debug_dir,
    #                         gt_label=label.data.long().numpy(),
    #                         gt_box=gt_boxes, threshold=threshold, gcam=gcam, g2=g2)

    return deterr_1, locerr_1, deterr_5, locerr_5, top_maps, top5_boxes

def calc_sim_map(root_maps, logit_maps, child_maps):
    pass

def val(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    with open(args.test_box, 'r') as f:
        gt_boxes = [list(map(float, x.strip().split(' ')[2:])) for x in f.readlines()]
    gt_boxes = [(box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1) for box in gt_boxes]

    # meters
    top1_clsacc = AverageMeter()
    top5_clsacc = AverageMeter()
    top1_clsacc.reset()
    top5_clsacc.reset()
    loc_err = {}
    for th in args.threshold:
        loc_err['top1_locerr_{}'.format(th)] = AverageMeter()
        loc_err['top1_locerr_{}'.format(th)].reset()
        loc_err['top5_locerr_{}'.format(th)] = AverageMeter()
        loc_err['top5_locerr_{}'.format(th)].reset()
        loc_err['top1_deterr_{}'.format(th)] = AverageMeter()
        loc_err['top1_deterr_{}'.format(th)].reset()
        loc_err['top5_deterr_{}'.format(th)] = AverageMeter()
        loc_err['top5_deterr_{}'.format(th)].reset()
        if args.eval_gcam:
            loc_err['top1_locerr_gcam_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_gcam_{}'.format(th)].reset()
            loc_err['top5_locerr_gcam_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_gcam_{}'.format(th)].reset()
            loc_err['top1_deterr_gcam_{}'.format(th)] = AverageMeter()
            loc_err['top1_deterr_gcam_{}'.format(th)].reset()
            loc_err['top5_deterr_gcam_{}'.format(th)] = AverageMeter()
            loc_err['top5_deterr_gcam_{}'.format(th)].reset()
        if args.eval_g2:
            loc_err['top1_locerr_g2_{}'.format(th)] = AverageMeter()
            loc_err['top1_locerr_g2_{}'.format(th)].reset()
            loc_err['top5_locerr_g2_{}'.format(th)] = AverageMeter()
            loc_err['top5_locerr_g2_{}'.format(th)].reset()
            loc_err['top1_deterr_g2_{}'.format(th)] = AverageMeter()
            loc_err['top1_deterr_g2_{}'.format(th)].reset()
            loc_err['top5_deterr_g2_{}'.format(th)] = AverageMeter()
            loc_err['top5_deterr_g2_{}'.format(th)].reset()
    # get model
    model = get_model(args)
    model.eval()

    # get data
    _, valcls_loader, valloc_loader = data_loader(args, test_path=True)
    assert len(valcls_loader) == len(valloc_loader), \
        'Error! Different size for two dataset: loc({}), cls({})'.format(len(valloc_loader), len(valcls_loader))

    # testing
    if args.debug:
        # show_idxs = np.arange(20)
        np.random.seed(2333)
        show_idxs = np.arange(len(valcls_loader))
        np.random.shuffle(show_idxs)
        show_idxs = show_idxs[:30]

    # evaluation classification task

    for idx, (dat_cls, dat_loc ) in tqdm(enumerate(zip(valcls_loader, valloc_loader))):
        grads = {}
        # parse data
        img_path, img, label_in = dat_cls
        if args.tencrop == 'True':
            bs, ncrops, c, h, w = img.size()
            img = img.view(-1, c, h, w)


        # forward pass
        args.device = torch.device('cuda') if args.gpus[0]>=0 else torch.device('cpu')
        img = img.to(args.device)
        # img_var, label_var = Variable(img), Variable(label)
        logits = model(img)
        root_logits = torch.mean(torch.mean(torch.mean(logits[0], dim=2), dim=2),dim=2)
        parent_logits = torch.mean(torch.mean(torch.mean(logits[1], dim=2), dim=2),dim=2)
        child_logits = torch.mean(torch.mean(torch.mean(logits[2], dim=2), dim=2),dim=2)
        logits = (root_logits, parent_logits, child_logits)
        # get classification prob
        logit3 = logits[-1]
        # logit3 = logits[-2]
        if args.bce or args.bbce and not args.lb:
        # if args.bce or args.bbce:
            logit3 = F.sigmoid(logit3)
        else:
            logit3 = F.softmax(logit3, dim=1)
        # if args.g2cam:
        #     grads[os.path.basename(img_path[0])].update(get_grad(model, logits0, last_layer_feats['cls5'], args.num_maps, layer='cls5'))
        if args.tencrop == 'True':
            logit3 = logit3.view(1, ncrops, -1).mean(1)
            if args.bbce and not args.lb:
                logit3 = logit3[:,:-args.sup]
        if args.NoHDA:
            logit2 = logit1 = [None]
        else:
            logit2 = logits[-2]
            if args.bce or args.bbce and not args.lb:
                logit2 = F.sigmoid(logit2)
            else:
                logit2 = F.softmax(logit2, dim=1)
            # if args.g2cam:
            #     grads[os.path.basename(img_path[0])].update(get_grad(model, logits1, last_layer_feats['cls4'], args.num_maps,layer='cls4'))
            if args.tencrop == 'True':
                logit2 = logit2.view(1, ncrops, -1).mean(1)
                if args.bbce and not args.lb:
                    logit2 = logit2[:,:-args.sup]

            logit1 = logits[-3]
            if args.bbce or args.bce and not args.lb:
                logit1 = F.sigmoid(logit1)
            else:
                logit1 = F.softmax(logit1, dim=1)
            # if args.g2cam:
            #     grads[os.path.basename(img_path[0])].update(get_grad(model, logits2, last_layer_feats['cls3'], args.num_maps, layer='cls3'))
            if args.tencrop == 'True':
                logit1 = logit1.view(1, ncrops, -1).mean(1)
                if args.bbce and not args.lb:
                    logit1 = logit1[:,:-args.sup]
        # update result record
        prec1_1, prec5_1 = evaluate.accuracy(logit3.cpu().data, label_in.long(), topk=(1, 5))
        top1_clsacc.update(prec1_1[0].numpy(), img.size()[0])
        top5_clsacc.update(prec5_1[0].numpy(), img.size()[0])

        # location
        _, img_loc, label = dat_loc
        img_loc = img_loc.to(args.device)
        # img_var, label_var = Variable(img), Variable(label)
        logits_loc = model(img_loc.requires_grad_())
        logits_ce = (logit1,logit2,logit3)
        if args.lb:
            logits_loc = logits_loc[:3]

        # if args.lb:
        #     child_maps = F.interpolate(model.module.get_child_maps_ce(), size=(28, 28), mode='bilinear',
        #                                align_corners=True)
        #     parent_maps = F.interpolate(model.module.get_parent_maps_ce(), size=(28, 28), mode='bilinear',
        #                                 align_corners=True)
        #     root_maps = model.module.get_root_maps_ce()
        # else:
        child_maps = F.interpolate(model.module.get_child_maps(), size=args.size, mode='bilinear', align_corners=True)
        parent_maps = F.interpolate(model.module.get_parent_maps(), size=args.size, mode='bilinear', align_corners=True)
        root_maps = model.module.get_root_maps()
        if args.bin_cls:
            bin_maps = model.module.get_bin_map()
        else:
            bin_maps = None
        if args.eval_gcam:
            last_layer_feats = model.module.last_layer_feat
            if args.NoHDA:
                logits_loc = logits_loc[-1:]
                logits_ce =(logit3,)
                layer_names = ['cls5']
                grads['gcam_cls4'] = grads['gcam_cls3'] = None
                grads['g2_cls4'] = grads['gcam_cls3'] = None
            else:
                layer_names = ['cls3', 'cls4', 'cls5']
            for logit, layer,logit_ce in zip(logits_loc,layer_names, logits_ce):
                if args.bce or args.bbce:
                    logit = F.sigmoid(logit)
                else:
                    logit = F.softmax(logit, dim=1)
                if args.bbce and not args.bg:
                    logit = logit[:,:-args.sup]
                grads.update(get_grad(model, logit, last_layer_feats[layer], args.num_maps, layer=layer, topk=(1,5),
                                      logits_ce=logit_ce, bg=args.bg))

        for th in args.threshold:
            df_deterr_1, df_locerr_1, df_deterr_5, df_locerr_5,df_top_maps, df_top5_boxes = eval_loc(logit3, logit2,
                                            logit1, child_maps, parent_maps, root_maps, img_path[0], args.input_size,
                                            args.crop_size, label, gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                                            debug=args.debug, debug_dir=args.debug_dir, NoHDA=args.NoHDA, bin_map = bin_maps)
            loc_err['top1_deterr_{}'.format(th)].update(df_deterr_1, img.size()[0])
            loc_err['top5_deterr_{}'.format(th)].update(df_deterr_5, img.size()[0])
            loc_err['top1_locerr_{}'.format(th)].update(df_locerr_1, img.size()[0])
            loc_err['top5_locerr_{}'.format(th)].update(df_locerr_5, img.size()[0])
            if args.debug and idx in show_idxs:
                sim_map = calc_sim_map(child_maps, parent_maps, root_maps)
                save_im_heatmap_box(img_path[0], df_top_maps, df_top5_boxes, args.debug_dir,
                                    gt_label=label.data.long().numpy(), sim_map=None,
                                    gt_box=gt_boxes[idx], epoch=args.current_epoch,threshold=th)
            if args.eval_gcam:
                # update result record
                deterr_1, locerr_1, deterr_5, locerr_5, top_maps, top5_boxes = eval_loc(logit3, logit2, logit1, grads['gcam_cls5'],grads['gcam_cls4'],
                                              grads['gcam_cls3'], img_path[0], args.input_size, args.crop_size,
                                              label, gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                                              debug=args.debug, debug_dir=args.debug_dir, gcam=True, NoHDA=args.NoHDA)
                loc_err['top1_deterr_gcam_{}'.format(th)].update(deterr_1, img.size()[0])
                loc_err['top5_deterr_gcam_{}'.format(th)].update(deterr_5, img.size()[0])
                loc_err['top1_locerr_gcam_{}'.format(th)].update(locerr_1, img.size()[0])
                loc_err['top5_locerr_gcam_{}'.format(th)].update(locerr_5, img.size()[0])
                if args.debug and th == 0.20:
                    if deterr_1 != df_deterr_1 or deterr_5 !=df_deterr_5:
                        save_im_heatmap_box(img_path[0], df_top_maps, df_top5_boxes, args.debug_dir,
                                            gt_label=label.data.long().numpy(),
                                            gt_box=gt_boxes[idx], threshold=th)
                        save_im_heatmap_box(img_path[0], top_maps, top5_boxes, args.debug_dir,
                                            gt_label=label.data.long().numpy(),
                                            gt_box=gt_boxes[idx], threshold=th, gcam=True, g2=False)
                        save_im_gcam_ggrads(img_path[0], grads, args.debug_dir, layers=['cls3', 'cls4', 'cls5'])

            if args.eval_g2:
                deterr_1, locerr_1, deterr_5, locerr_5,top_maps, top5_boxes = eval_loc(logit3, logit2, logit1, grads['g2_cls5'], grads['g2_cls4'],
                                              grads['g2_cls3'], img_path[0], args.input_size, args.crop_size,
                                              label, gt_boxes[idx], topk=(1, 5), threshold=th, mode='union',
                                              debug=args.debug, debug_dir=args.debug_dir, g2=True)
                loc_err['top1_deterr_g2_{}'.format(th)].update(deterr_1, img.size()[0])
                loc_err['top5_deterr_g2_{}'.format(th)].update(deterr_5, img.size()[0])
                loc_err['top1_locerr_g2_{}'.format(th)].update(locerr_1, img.size()[0])
                loc_err['top5_locerr_g2_{}'.format(th)].update(locerr_5, img.size()[0])

    print('== cls err')
    print('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
    for th in args.threshold:
        print('=========== threshold: {} ==========='.format(th))
        print('== det err')
        print('DANet-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_deterr_{}'.format(th)].avg,
                                                         loc_err['top5_deterr_{}'.format(th)].avg))
        if args.eval_gcam:
            print('GCAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_deterr_gcam_{}'.format(th)].avg,
                                                            loc_err['top5_deterr_gcam_{}'.format(th)].avg))
        if args.eval_g2:
            print('G2-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_deterr_g2_{}'.format(th)].avg,
                                                          loc_err['top5_deterr_g2_{}'.format(th)].avg))
        print('== loc err')
        print('DANet-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                         loc_err['top5_locerr_{}'.format(th)].avg))
        if args.eval_gcam:
            print('GCAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_gcam_{}'.format(th)].avg,
                                                            loc_err['top5_locerr_gcam_{}'.format(th)].avg))
        if args.eval_g2:
            print('G2-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_g2_{}'.format(th)].avg,
                                                          loc_err['top5_locerr_g2_{}'.format(th)].avg))
    result_log = os.path.join(args.snapshot_dir, 'results.log')
    with open(result_log,'a') as fw:
        fw.write('current_epoch:{}\n'.format(args.current_epoch))
        fw.write('== cls err ')
        fw.write('Top1: {:.2f} Top5: {:.2f}\n'.format(100.0 - top1_clsacc.avg, 100.0 - top5_clsacc.avg))
        for th in args.threshold:
            fw.write('=========== threshold: {} ===========\n'.format(th))
            fw.write('== det err ')
            fw.write('DANet-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_deterr_{}'.format(th)].avg,
                                                             loc_err['top5_deterr_{}'.format(th)].avg))
            if args.eval_gcam:
                fw.write('GCAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_deterr_gcam_{}'.format(th)].avg,
                                                                loc_err['top5_deterr_gcam_{}'.format(th)].avg))
            if args.eval_g2:
                fw.write('G2-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_deterr_g2_{}'.format(th)].avg,
                                                              loc_err['top5_deterr_g2_{}'.format(th)].avg))
            fw.write('== loc err ')
            fw.write('DANet-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_{}'.format(th)].avg,
                                                       loc_err['top5_locerr_{}'.format(th)].avg))
            if args.eval_gcam:
                fw.write('GCAM-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_gcam_{}'.format(th)].avg,
                                                           loc_err['top5_locerr_gcam_{}'.format(th)].avg))
            if args.eval_g2:
                fw.write('G2-Top1: {:.2f} Top5: {:.2f}\n'.format(loc_err['top1_locerr_g2_{}'.format(th)].avg,
                                                           loc_err['top5_locerr_g2_{}'.format(th)].avg))

if __name__ == '__main__':
    args = opts().parse()
    val(args)
