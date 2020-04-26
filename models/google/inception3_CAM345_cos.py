import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from torch.autograd import Variable
import os
import cv2
import numpy as np

__all__ = ['Inception3', 'model']

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def model(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        # if 'transform_input' not in kwargs:
        #     kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model_dict = model.state_dict()
        pretrained_file = os.path.join(kwargs['args'].pretrained_model_dir, kwargs['args'].pretrained_model)
        if os.path.isfile(pretrained_file):
            pretrained_dict = torch.load(pretrained_file)
            print('load pretrained model from {}'.format(pretrained_file))
        else:
            pretrained_dict = model_zoo.load_url(model_urls['inception_v3_google'])
            print('load pretrained model from: {}'.format(model_urls['inception_v3_google']))
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from inception v3'.format(k))
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):
    def __init__(self, num_classes=1000, args=None, threshold=0.6, transform_input=False):
        super(Inception3, self).__init__()
        # ====================== network settings ==============================
        self.num_classes = num_classes
        self.threshold = threshold
        self.transform_input = transform_input
        self.cos_alpha = args.cos_alpha
        self.num_maps = 4
        self.root_num_classes = 11
        self.parent_num_classes = 37
        self.child_num_classes = num_classes
        self.args = args
        self.sup_root_classes = 11 + args.sup if args.bbce else 11
        self.sup_parent_classes = 37 + args.sup if args.bbce else 37
        self.sup_child_classes = num_classes + args.sup if args.bbce else num_classes
        # ====================== backbone ==============================
        # original inception_v3 layers
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)  # spatial scale half
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)  # spatial scale minus 1
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)  # spatial scale minus 1
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # ============================= added layers ==================================
        self.classifier4 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(256, self.sup_root_classes*self.num_maps, kernel_size=1, padding=0)
        )

        self.classifier5 = nn.Sequential(
            nn.Conv2d(288, 384, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(384, self.sup_parent_classes*self.num_maps, kernel_size=1, padding=0)
        )

        self.classifier6 = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, self.sup_child_classes*self.num_maps, kernel_size=1, padding=0)
        )
        if self.args.lb:
            self.classifier6_ce = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, num_classes*self.num_maps, kernel_size=1, padding=0)
            )


        # ================================ loss ===================================
        self.loss_cross_entropy = F.cross_entropy
        self.loss_bce = F.binary_cross_entropy_with_logits
        # =============================== initialize ==============================
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, label=None):

        batch_size = x.shape[0]

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        self.root_map = self.classifier4(x)


        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        self.parent_map = self.classifier5(x)


        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        features = self.Mixed_6e(x)

        # ============================= classifier_1 ============================
        cam = self.classifier6(features)

        self.child_map = cam
        if self.args.lb:
            cam_ce = self.classifier6_ce(features)
            self.child_map_ce = cam_ce

        self.root_map = self.root_map.view(batch_size, self.sup_root_classes, self.num_maps, 25, 25)
        self.parent_map = self.parent_map.view(batch_size, self.sup_parent_classes, self.num_maps, 25, 25)
        self.child_map = self.child_map.view(batch_size, self.sup_child_classes, self.num_maps, 12, 12)
        if self.args.lb:
            self.child_map_ce = self.child_map_ce.view(batch_size, self.num_classes, self.num_maps, 12, 12)

        root_logits = torch.mean(torch.mean(torch.mean(self.root_map, dim=2), dim=2), dim=2)
        parent_logits = torch.mean(torch.mean(torch.mean(self.parent_map, dim=2), dim=2), dim=2)
        child_logits = torch.mean(torch.mean(torch.mean(self.child_map, dim=2), dim=2), dim=2)
        if self.args.lb:
            child_logits_ce = torch.mean(torch.mean(torch.mean(self.child_map_ce, dim=2), dim=2), dim=2)
        if self.args.lb:
            return root_logits, parent_logits, child_logits, child_logits_ce
        else:
            return root_logits, parent_logits, child_logits


    def calculate_cosineloss(self, maps):

        batch_size = maps.size(0)
        num_maps = maps.size(1)
        if self.args.NoHDA:
            channel_num = int(self.num_maps/2)
        else:
            channel_num = int(self.num_maps*3/2)
        eps = 1e-8
        random_seed = random.sample(range(num_maps), channel_num)
        maps = maps[:, random_seed, :, :].view(batch_size, channel_num, -1)

        X1 = maps.unsqueeze(1)
        X2 = maps.unsqueeze(2)
        dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
        # print(dot12)
        dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))
        tri_tensor = ((torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1]*channel_num))).expand(batch_size, channel_num, channel_num)).cuda()

        dist_num = abs((tri_tensor*dist).sum(1).sum(1)).sum()/(batch_size*channel_num*(channel_num-1)/2)

        return dist_num, random_seed

    def get_gt_map(self, gt_root_label, gt_parent_label, gt_child_label):
        batch_size = self.child_map.size(0)
        child_map = self.child_map.reshape(batch_size*self.sup_child_classes, self.num_maps, 12, 12).clone()
        child_map = child_map[[gt_child_label[i].long()+(i*self.sup_child_classes) for i in range(batch_size)], :, :, :]
        child_map = F.interpolate(child_map.reshape(batch_size, self.num_maps, 12, 12),size=(25, 25), mode='bilinear', align_corners=True)

        if self.args.NoHDA:
            return child_map

        parent_map = self.parent_map.reshape(batch_size * self.sup_parent_classes, self.num_maps, 25, 25).clone()
        parent_map = parent_map[[gt_parent_label[i].long() + (i * self.sup_parent_classes) for i in range(batch_size)], :, :, :]
        parent_map = parent_map.reshape(batch_size, self.num_maps, 25, 25)

        root_map = self.root_map.reshape(batch_size * self.sup_root_classes, self.num_maps, 25, 25).clone()
        root_map = root_map[[gt_root_label[i].long() + (i * self.sup_root_classes) for i in range(batch_size)], :, :, :]
        root_map = root_map.reshape(batch_size, self.num_maps, 25, 25)

        return torch.cat((child_map,parent_map, root_map),1)

    def get_gt_map_ce(self, gt_root_label, gt_parent_label, gt_child_label):
        batch_size = self.child_map_ce.size(0)
        child_map = self.child_map_ce.reshape(batch_size*self.child_num_classes, self.num_maps, 12, 12).clone()
        child_map = child_map[[gt_child_label[i].long()+(i*self.child_num_classes) for i in range(batch_size)], :, :, :]
        child_map = F.interpolate(child_map.reshape(batch_size, self.num_maps, 12, 12),size=(25, 25), mode='bilinear', align_corners=True)

        if self.args.NoHDA:
            return child_map

    def get_loss(self, logits, gt_root_label, gt_parent_label, gt_child_label, child_h=None, child_v=None, parent_h=None, parent_v=None,
                 root_h=None, root_v=None):
        if self.args.lb:
            root_logits, parent_logits, child_logits, child_logits_ce = logits
        else:
            root_logits, parent_logits, child_logits = logits

        maps = self.get_gt_map(gt_root_label, gt_parent_label, gt_child_label)
        # maps = torch.cat((
        #     F.upsample((self.child_map.reshape(batch_size*self.num_classes, self.num_maps, 12, 12)[[gt_child_label[i].long()+(i*self.num_classes) for i in range(batch_size)], :, :, :]).reshape(batch_size, self.num_maps, 12, 12),size=(25, 25), mode='bilinear', align_corners=True),
        #     F.upsample((self.parent_map.reshape(batch_size * 37, self.num_maps, 25, 25)[[parent_logits[i].long() + (i * 37) for i in range(batch_size)], :, :, :]).reshape(batch_size, self.num_maps, 25, 25), size=(25, 25),
        #                mode='bilinear', align_corners=True),
        #     (self.root_map.reshape(batch_size*11, self.num_maps, 25, 25)[[root_logits[i].long() + (i * 11) for i in range(batch_size)], :, :, :]).reshape(batch_size, self.num_maps, 25, 25)), 1)

        loss_cos, random_seed = self.calculate_cosineloss(maps)
        root_loss_cls = torch.zeros_like(loss_cos)
        parent_loss_cls = torch.zeros_like(loss_cos)
        if self.args.lb:
            maps_ce = self.get_gt_map_ce(gt_root_label, gt_parent_label, gt_child_label)
            loss_cos_ce, _ = self.calculate_cosineloss(maps_ce)

            assert self.args.NoHDA, 'Only work without HDA'
            # mce
            child_loss_cls_ce = self.loss_cross_entropy(child_logits_ce, gt_child_label.long())
            # bbce
            n = child_logits.size(0)
            gt_child = torch.zeros_like(child_logits)
            gt_child[:, -self.args.sup:] = 1
            gt_child[torch.arange(n), gt_child_label.long()] = 1
            child_loss_cls = self.loss_bce(child_logits, gt_child)

            loss_val = 0.5 * child_loss_cls_ce + self.cos_alpha * loss_cos_ce \
                       +0.5 * child_loss_cls + self.cos_alpha * loss_cos
        elif self.args.mce:
            root_loss_cls = self.loss_cross_entropy(root_logits, gt_root_label.long())
            parent_loss_cls = self.loss_cross_entropy(parent_logits, gt_parent_label.long())
            child_loss_cls = self.loss_cross_entropy(child_logits, gt_child_label.long())

            loss_val = 0.5 * root_loss_cls + 0.5 * parent_loss_cls + 0.5 * child_loss_cls + self.cos_alpha * loss_cos
            if self.args.NoHDA:
                loss_val = 0.5 * child_loss_cls + self.cos_alpha * loss_cos
        else:
            assert self.args.bbce, 'args.bbce should be true.'
            n = root_logits.size(0)
            gt_root = torch.zeros_like(root_logits)
            gt_root[:, -self.args.sup:] = 1
            gt_root[torch.arange(n), gt_root_label.long()] = 1

            gt_parent = torch.zeros_like(parent_logits)
            gt_parent[:, -self.args.sup:] = 1
            gt_parent[torch.arange(n), gt_parent_label.long()] = 1

            gt_child = torch.zeros_like(child_logits)
            gt_child[:, -self.args.sup:] = 1
            gt_child[torch.arange(n), gt_child_label.long()] = 1

            root_loss_cls = self.loss_bce(root_logits, gt_root)
            parent_loss_cls = self.loss_bce(parent_logits, gt_parent)
            child_loss_cls = self.loss_bce(child_logits, gt_child)
            if self.args.NoHDA:
                loss_val = 0.5 * child_loss_cls + self.cos_alpha * loss_cos
            else:
                loss_val = 0.5 * root_loss_cls + 0.5 * parent_loss_cls + 0.5 * child_loss_cls + self.cos_alpha * loss_cos

        loss_sim = torch.zeros_like(loss_val)
        return loss_val, root_loss_cls, parent_loss_cls, child_loss_cls, loss_cos, loss_sim


    def get_child_maps(self):
        return torch.mean(F.relu(self.child_map), dim=2)

    def get_parent_maps(self):
        return torch.mean(F.relu(self.parent_map), dim=2)

    def get_root_maps(self):
        return torch.mean(F.relu(self.root_map), dim=2)



class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        self.stride = stride
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=stride)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
