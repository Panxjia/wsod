import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
import os

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'model'
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, cnvs=(10,17,24), args=None):
        super(VGG, self).__init__()
        self.conv1_2 = nn.Sequential(*features[:cnvs[0]])
        self.conv3 = nn.Sequential(*features[cnvs[0]:cnvs[1]])
        self.conv4 = nn.Sequential(*features[cnvs[1]:cnvs[2]])
        # self.conv1_4 = nn.Sequential(*features[:-5])
        self.conv5 = nn.Sequential(*features[cnvs[2]:-1])
        self.fmp = features[-1]  # final max pooling
        self.num_classes = num_classes
        self.args = args
        self.cls_feat = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True)
        )
        self.cls = nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0)

        if self.args.loc_branch:
            self.loc = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
                nn.Conv2d(1024, 1, kernel_size=1, padding=0)
            )
            # self.loc = nn.Sequential(
            #     nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            #     nn.ReLU(True),
            #     nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1),
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(1024, self.num_classes + 1, kernel_size=1, padding=0)
            # )
        # if self.args.com_feat:
        #     self.conv35 = nn.Sequential(
        #         nn.Conv2d(256, 128, kernel_size=1, padding=0, dilation=1),  # fc6
        #     )
        #     self.conv45 = nn.Sequential(
        #         nn.Conv2d(512, 128, kernel_size=1, padding=0, dilation=1),  # fc6
        #     )
        #     self.conv55 = nn.Sequential(
        #         nn.Conv2d(512, 128, kernel_size=1, padding=0, dilation=1),  # fc6
        #     )
        #
        #     self.merge = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Conv2d(128, 512, kernel_size=1, padding=0, dilation=1),
        #         nn.ReLU()
        #     )
        if self.args.non_local:
            if '3' in self.args.nl_blocks:
                self.nl_phi3 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1),
                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True)
                    )
                self.nl_theta3 = nn.Sequential(nn.Conv2d(256,128,kernel_size=1),
                                             nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True)
                    )
            if '4' in self.args.nl_blocks:
                self.nl_phi4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True)
                    )
                self.nl_theta4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True)
                    )
            if '5' in self.args.nl_blocks:
                self.nl_phi5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True)
                    )
                self.nl_theta5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True)
                    )
        if self.args.memo:
            self.memo_module = memomy_module_v0(num_classes, fg_th=self.args.th_fg, bg_th=self.args.th_bg,
                                                lr=self.args.memo_lr)
        if self.args.bifpn:
            self.fpn_mix4_1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3,  padding=1)
            )
            self.fpn_mix3_2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.fpn_mix4_2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.fpn_mix5_2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

            self.fpn_f4_0_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
            self.fpn_f3_0_2 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
            self.fpn_f4_0_2 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
            self.fpn_f4_1_2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
            self.fpn_f5_0_2 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
            self.fpn_f5_0_41 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
            self.fpn_f3_2_down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fpn_f4_2_down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        elif self.args.fpn:
            self.fpn_lat_3 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
            self.fpn_lat_4 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
            self.fpn_lat_5 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
            self.fpn_out_3 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.fpn_out_3_exp = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.fpn_out_4 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.fpn_out_5 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            pass

        self._initialize_weights()

        # loss function
        self.loss_cross_entropy = F.cross_entropy
        self.loss_bce = F.binary_cross_entropy_with_logits
        self.nll_loss = F.nll_loss

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



    def forward(self, x, is_training=False, label=None, erase_start=False):
        x = self.conv1_2(x)
        feat_3 = self.conv3(x)
        if self.args.non_local and '3' in self.args.nl_blocks:
            feat3_phi = self.nl_phi3(feat_3)
            feat3_theta = self.nl_theta3(feat_3)
            feat_3 = self.non_local(feat_3, feat3_phi, feat3_theta, self.args.nl_kernel)

        self.feat3 = feat_3
        feat_4 = self.conv4(feat_3)
        if self.args.non_local and '4' in self.args.nl_blocks:
            feat4_phi = self.nl_phi4(feat_4)
            feat4_theta = self.nl_theta4(feat_4)
            feat_4 = self.non_local(feat_4,feat4_phi, feat4_theta, self.args.nl_kernel)
        self.feat4 = feat_4
        feat_5 = self.conv5(feat_4)
        if self.args.l5_red:
            feat_5 = self.fmp(feat_5)
        if self.args.non_local and '5' in self.args.nl_blocks:
            feat5_phi = self.nl_phi5(feat_5)
            feat5_theta = self.nl_theta5(feat_5)
            feat_5 = self.non_local(feat_5, feat5_phi, feat5_theta, self.args.nl_kernel)
        self.feat5 = feat_5

        if self.args.fpn or self.args.bifpn:
            feat_3_2, feat_4_2, feat_5_2 = self.construct_fpn(feat_3, feat_4, feat_5)
            if self.args.erase and is_training and erase_start:
                ## top-down
                feat_5_cls = self.cls_feat(feat_5_2)
                cls_map = self.cls_feat(feat_5_cls)
                erase_mask_54 = self.erase_map(cls_map, feat_4_2, label, erase_th=self.args.erase_th_l5, var=self.args.var_erase)
                feat_4_2 = feat_4_2 * erase_mask_54.detach()
                self.erase_mask_54 = erase_mask_54
                if self.args.neg_erase:
                    neg_mask_4, neg_mask_3 = self.neg_map(cls_map, feat_4_2, feat_3_2)
                    feat_4_2 = feat_4_2 * neg_mask_4.detach()
                feat_4_cls = self.cls_feat(feat_4_2)
                cls_map_4 = self.cls(feat_4_cls)
                erase_mask_43 = self.erase_map(cls_map_4, feat_3_2, label, erase_th=self.args.erase_th_l4,
                                              erased_mask=erase_mask_54, var=self.args.var_erase)
                feat_3_2 = feat_3_2 * erase_mask_43.detach()
                self.erase_mask_43 = erase_mask_43
                if self.args.neg_erase:
                    feat_3_2 = feat_3_2 * neg_mask_3.detach()
                feat_3_cls = self.cls_feat(feat_3_2)
                cls_map_3 = self.cls(feat_3_cls)
                ### bottom-up
                # cls_map_3 = self.cls3(feat_3_2)
                # erase_mask_34 = self.erase_map(cls_map_3, feat_4_2, label, erase_th=self.args.erase_th_l3)
                # feat_4_2 = feat_4_2 * erase_mask_34.detach()
                # if self.args.neg_erase:
                #     neg_mask_4, neg_mask_5 = self.neg_map(cls_map_3, feat_4_2, feat_5_2)
                #     feat_4_2 = feat_4_2 * neg_mask_4.detach()
                # cls_map_4 = self.cls4(feat_4_2)
                # erase_map_45 = self.erase_map(cls_map_4, feat_5_2, label, erase_th=self.args.erase_th_l4,
                #                               erased_mask=erase_mask_34)
                # feat_5_2 = feat_5_2 * erase_map_45.detach()
                # cls_map = self.cls(feat_5_2)
                # if self.args.neg_erase:
                #     feat_5_2 = feat_5_2 * neg_mask_5.detach()

            else:
                # cls_map = self.cls(feat_5_2)
                # cls_map_4 = self.cls4(feat_4_2)
                # cls_map_3 = self.cls3(feat_3_2)
                feat_5_cls = self.cls_feat(feat_5_2)
                feat_4_cls = self.cls_feat(feat_4_2)
                feat_3_cls = self.cls_feat(feat_3_2)
                cls_map = self.cls(feat_5_cls)
                cls_map_4 = self.cls(feat_4_cls)
                cls_map_3 = self.cls(feat_3_cls)

            self.feat_5_cls = feat_5_cls
            self.feat_4_cls = feat_4_cls
            self.feat_3_cls = feat_3_cls

            self.cls_map = cls_map
            self.cls_map_4 = cls_map_4
            self.cls_map_3 = cls_map_3
            if self.args.loc_branch:
                if self.args.loc_layer == 3:
                    feat_loc = feat_3_2
                elif self.args.loc_layer == 4:
                    feat_loc = feat_4_2
                else:
                    feat_loc = feat_5_2
                loc_map = self.loc(feat_loc)
                self.loc_map = loc_map
        else:
            cls_map = self.cls(feat_5)
            self.cls_map = cls_map
            if self.args.loc_branch:
                loc_map = self.loc(feat_5)
                self.loc_map = loc_map
        return cls_map

    def norm_atten_map(self, map):
        min_val, _ = torch.min(torch.min(map, dim=-1, keepdim=True)[0], dim=-1, keepdim=True)
        max_val, _ = torch.max(torch.max(map, dim=-1, keepdim=True)[0], dim=-1, keepdim=True)
        norm_map = (map - min_val) / (max_val - min_val + 1e-15)
        return norm_map

    def erase_map(self, feat_in, feat_out, label, erase_th=0.5, erased_mask=None, var=False):
        n, c, hl, wl = feat_out.size()
        if var:
            cls_logits = F.softmax(feat_in, dim=1)
            var_logits = torch.var(cls_logits, dim=1).squeeze()
            norm_var_logits = self.norm_atten_map(var_logits)
            earse_mask = norm_var_logits >= erase_th
        else:
            atten_map_h = feat_in[torch.arange(n), label.long(), ...]
            norm_atten_map_h = self.norm_atten_map(atten_map_h.clone())
            earse_mask = norm_atten_map_h >= erase_th
        earse_mask = earse_mask.float().unsqueeze(1).repeat(1, c, 1, 1)
        erase_mask = F.interpolate(earse_mask, size=(hl, wl), mode='nearest')

        if erased_mask is not None:
            erased_mask = erased_mask[:,0,...].unsqueeze(1).repeat(1, c, 1, 1)
            erased_mask = F.interpolate(erased_mask, size=(hl, wl), mode='nearest')
            return erase_mask * erased_mask
        return erase_mask

    def neg_map(self, logits, logits_l1, logits_l2, neg_th=0.2):
        n, c1, h1, w1 = logits_l1.size()
        n, c2, h2, w2 = logits_l2.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1).squeeze()
        norm_var_logits_5 = self.norm_atten_map(var_logits)
        neg_mask = norm_var_logits_5 > neg_th
        neg_mask_1 = neg_mask.float().unsqueeze(1).repeat(1,c1,1,1)
        neg_mask_2 = neg_mask.float().unsqueeze(1).repeat(1,c2,1,1)
        neg_mask_1 = F.interpolate(neg_mask_1, size=(h1,w1), mode='nearest')
        neg_mask_2 = F.interpolate(neg_mask_2, size=(h2,w2), mode='nearest')

        return neg_mask_1, neg_mask_2

    def construct_fpn(self, f3_0, f4_0, f5_0):
        f3_2 = f4_2 = f5_2 = None
        if self.args.bifpn:
            """
            F5_0  ------------------------> F5_2 ------>
                | ------------ |             ^
                               V             |
            F4_0  ----------  F4_1 -------- F4_2 ------>
                | ------------ | ------------^^
                               | ----------- V
            F3_0 -------------------------- F3_2 ------>
            :return: F3_2, F4_2, F5_2

            """

            f5_0_41 = self.fpn_f5_0_41(f5_0)
            self.fpn_f5_0_41_up = F.interpolate(f5_0_41, scale_factor=2, mode='bilinear', align_corners=True)

            f4_1 = self.fpn_mix4_1(self.fpn_f5_0_41_up + self.fpn_f4_0_1(f4_0))
            self.f4_1_up = F.interpolate(f4_1, scale_factor=2, mode='bilinear', align_corners=True)

            f3_2 = self.fpn_mix3_2(self.fpn_f3_0_2(f3_0) + self.f4_1_up)
            f4_2 = self.fpn_mix4_2(self.fpn_f4_0_2(f4_0) + self.fpn_f4_1_2(f4_1) + self.fpn_f3_2_down(f3_2))
            f5_2 = self.fpn_mix5_2(self.fpn_f5_0_2(f5_0) + self.fpn_f4_2_down(f4_2))

        elif self.args.fpn:
            h3, w3 = f3_0.size()[2:]
            h4, w4 = f4_0.size()[2:]
            lateral_5_conv = self.fpn_lat_5(f5_0)
            f5_2 = self.fpn_out_5(lateral_5_conv)+ f5_0
            f5_up = F.interpolate(lateral_5_conv, size=(h4, w4), mode='bilinear', align_corners=True)
            lateral_4_conv = self.fpn_lat_4(f4_0) + f5_up
            f4_2 = self.fpn_out_4(lateral_4_conv) + f4_0
            f4_up = F.interpolate(lateral_4_conv, size=(h3, w3), mode='bilinear', align_corners=True)
            f3_2 = self.fpn_out_3_exp(self.fpn_out_3(self.fpn_lat_3(f3_0) + f4_up) + f3_0)
        else:
            pass

        return f3_2, f4_2, f5_2


    def non_local(self, feat, f_phi, f_theta,kernel=3):
        n, c, h, w = feat.size()
        c_nl = f_phi.size(1)
        f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
        f_theta = f_theta.contiguous().view(n, c_nl, -1)
        non_local_cos = torch.bmm(f_phi, f_theta)
        feat_neighbor_mat = torch.zeros_like(non_local_cos)
        if kernel >0:
            feat_neighbor_loc = self.neighbor_area(h, w, kernel=kernel)
            ind = torch.arange(int(h*w)).view(-1,1)
            ind = ind.repeat(1,kernel*kernel)
            feat_neighbor_mat[:,ind,feat_neighbor_loc] = 1
            ## pairwise function
            # 1. -- guassian embedding--
            if self.args.nl_pairfunc == 0:
                non_local_cos[feat_neighbor_mat < 1] = -1e15
                non_local_cos = non_local_cos.clamp(min=-1e20, max=1e20)
                non_local_cos = F.softmax(non_local_cos, dim=2)

            # 2. -- dot production--
            elif self.args.nl_pairfunc == 1:
                non_local_cos[feat_neighbor_mat < 1] = 0
                norm_n = feat_neighbor_mat.sum(dim=-1, keepdim=True)
                non_local_cos = non_local_cos / norm_n
            else:
                print('Wrong value of non local pairwise function.')
        else:
            ## pairwise function
            # 1. -- guassian embedding--
            if self.args.nl_pairfunc == 0:
                non_local_cos = non_local_cos.clamp(min=-1e20, max=1e20)
                non_local_cos = F.softmax(non_local_cos, dim=2)

            # 2. -- dot production--
            elif self.args.nl_pairfunc == 1:
                norm_n = torch.ones_like(non_local_cos)*(h*w)
                non_local_cos = non_local_cos/norm_n
            else:
                print('Wrong value of non local pairwise function.')

        feat_nl = feat.permute(0, 2, 3, 1).contiguous().view(n, -1, c)
        feat_nl = torch.bmm(non_local_cos, feat_nl)
        feat_nl = feat_nl.contiguous().view(n, h, w, c).permute(0, 3, 1, 2)
        if torch.isnan(feat_nl).sum()>0:
            print('fuck.')
        if self.args.nl_residual:
            return feat_nl+feat
        else:
            return feat_nl

    def neighbor_area(self, h, w, kernel=3):
        feat_x_axis = torch.arange(0,h)
        feat_y_axis = torch.arange(0,w)
        feat_x_coor, feat_y_coor = torch.meshgrid(feat_x_axis,feat_y_axis)
        feat_x_coor = feat_x_coor.contiguous().view(-1,1)
        feat_y_coor = feat_y_coor.contiguous().view(-1,1)
        xy_range = (kernel-1)//2
        neighbor_x_axis = torch.arange(-xy_range,xy_range+1)
        neighbor_y_axis = torch.arange(-xy_range,xy_range+1)
        neighbor_x_coor, neighbor_y_coor = torch.meshgrid(neighbor_x_axis, neighbor_y_axis)
        neighbor_x_coor = neighbor_x_coor.contiguous().view(-1)
        neighbor_y_coor = neighbor_y_coor.contiguous().view(-1)
        feat_neigh_x_coor = feat_x_coor + neighbor_x_coor
        feat_neigh_y_coor = feat_y_coor + neighbor_y_coor
        feat_neigh_x_coor = feat_neigh_x_coor.clamp(min=0,max=h-1)
        feat_neigh_y_coor = feat_neigh_y_coor.clamp(min=0,max=w-1)
        feat_neigh_loc = feat_neigh_x_coor * w + feat_neigh_y_coor
        return feat_neigh_loc

    def get_cls_simliar_loss(self, gt_label, maps_h, maps_v, cls_prot_h, cls_prot_v):
        cls_ids = torch.unique(gt_label)
        loss = 0
        for cls_i in cls_ids:
            ind_i = gt_label == cls_i
            map_cls_h = maps_h[ind_i,cls_i,:]
            map_cls_v = maps_v[ind_i,cls_i,:]

            loss += ((map_cls_h - cls_prot_h[cls_i,:])**2).sum()
            loss += ((map_cls_v - cls_prot_v[cls_i, :])**2).sum()
        loss = loss /len(gt_label)
        return loss


    def get_loss(self, logits, gt_child_label, protype_h=None, protype_v=None, epoch=0, loc_start=10, erase_start=False):
        cls_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
        loss = 0
        loss += self.args.loss_w_5 * self.loss_cross_entropy(cls_logits, gt_child_label.long())
        if self.args.fpn or self.args.bifpn:
            cls_logits_4 = torch.mean(torch.mean(self.cls_map_4, dim=2), dim=2)
            cls_logits_3 = torch.mean(torch.mean(self.cls_map_3, dim=2), dim=2)
            loss_4 = self.args.loss_w_4 * self.loss_cross_entropy(cls_logits_4, gt_child_label.long())
            loss += loss_4
            loss_3 = self.args.loss_w_3 * self.loss_cross_entropy(cls_logits_3, gt_child_label.long())
            loss += loss_3
        if self.args.memo:
            if self.args.loc_layer == 3:
                memo_feat = self.feat_3_cls
                memo_logits = self.cls_map_3
            elif self.args.loc_layer == 4:
                memo_feat = self.feat_4_cls
                memo_logits = self.cls_map_4
            else:
                memo_feat = self.feat_5_cls
                memo_logits = logits
            memo_kv = self.memo_module.get_kv(memo_feat.detach(), memo_logits, self.loc_map, gt_child_label)
            self.memo_module.update(memo_kv, epoch)

        if self.args.loc_branch and epoch >= loc_start:
            if self.args.loc_layer == 3:
                loc_logits = self.cls_map_3
            elif self.args.loc_layer == 4:
                loc_logits = self.cls_map_4
            else:
                loc_logits = logits

            loc_loss = self.get_loc_loss(loc_logits, gt_child_label, self.args.th_bg, self.args.th_fg)
            loss += self.args.loss_w_loc * loc_loss
        else:
            loc_loss = torch.zeros_like(loss)
        return loss, loc_loss

    def get_loc_loss(self, logits, gt_child_label, th_bg=0.3, th_fg=0.5):
        if self.args.memo:
            loc_map_cls, cls_mask, bin_weight = self.gene_loc_gt_memo(gt_child_label, alpha=self.args.memo_alpha,
                                                                      beta=self.args.memo_beta)
        else:
            loc_map_cls, cls_mask, bin_weight = self.gene_loc_gt(logits, gt_child_label, th_bg, th_fg)

        bin_loss = self.loss_bce(loc_map_cls, cls_mask, reduction='none')
        bin_loss = bin_loss * bin_weight
        return torch.sum(bin_loss) / torch.sum(bin_weight)

        # loss = self.loss_cross_entropy(self.loc_map, gt_child_label_cp.long(), reduction='none')
        # loss = torch.sum(loss * pos_sam)
        # return loss/float(torch.sum(pos_sam))

    def gene_loc_gt(self, logits, gt_child_label, th_bg=0.3, th_fg=0.5):
        n, c, lh, lw = self.loc_map.size()
        if self.args.avg_bin:
            loc_map_cls = F.avg_pool2d(self.loc_map, kernel_size=self.args.avg_size, stride=self.args.avg_stride)
            # loc_map_cls = F.adaptive_avg_pool2d(self.loc_map, output_size=1)
        else:
            loc_map_cls = self.loc_map

        _, _, h, w = logits.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = normalize_feat(var_logits)
        norm_var_logits = F.interpolate(norm_var_logits.unsqueeze(1), size=(lh, lw), mode='bilinear',
                                        align_corners=True)

        fg_cls = cls_logits[torch.arange(n), gt_child_label.long(), ...].clone()
        fg_cls = normalize_feat(fg_cls)
        norm_fg_cls = F.interpolate(fg_cls.unsqueeze(1), size=(lh, lw), mode='bilinear', align_corners=True)
        # norm_fg_cls = norm_fg_cls.squeeze()

        # norm_cls = (norm_var_logits + norm_fg_cls) * 0.5
        cls_mask = -1 * torch.ones_like(norm_var_logits)
        cls_mask[norm_var_logits < th_bg] = 0.
        cls_mask[norm_fg_cls > th_fg] = 1.

        if self.args.avg_bin:
            cls_mask = F.max_pool2d(cls_mask, kernel_size=self.args.avg_size, stride=self.args.avg_stride)
            # cls_mask = F.adaptive_max_pool2d(cls_mask,output_size=1)
        if self.args.adap_w:
            # bin_weight = torch.exp(self.args.adap_w_gama*torch.abs(norm_var_logits - norm_fg_cls))
            loc_logits = torch.sigmoid(loc_map_cls)
            bin_weight_pos = cls_mask * (torch.pow((1 - loc_logits), self.args.adap_w_gama))
            bin_weigt_neg = (1 - cls_mask) * (torch.pow(loc_logits, self.args.adap_w_gama))
            bin_weight = bin_weight_pos + bin_weigt_neg
            bin_weight[cls_mask < 0] = 0.

        else:
            bin_weight = torch.ones_like(cls_mask)
            bin_weight[cls_mask < 0] = 0.

        return loc_map_cls, cls_mask, bin_weight

    def gene_loc_gt_memo(self, gt_child_label, alpha=1., beta=1.0):
        if self.args.avg_bin:
            loc_map_cls = F.avg_pool2d(self.loc_map, kernel_size=self.args.avg_size, stride=self.args.avg_stride)
            # loc_map_cls = F.adaptive_avg_pool2d(self.loc_map, output_size=1)
        else:
            loc_map_cls = self.loc_map
        cls_feat = self.memo_module.feat_cls_cur

        memo_neg_feat = self.memo_module.get_v(200)
        memo_pos_feat = self.memo_module.get_v(list(gt_child_label.long().cpu().numpy()))

        n, c, h, w = cls_feat.size()
        cls_feat_tmp = cls_feat.clone()
        cls_feat_tmp  = cls_feat_tmp.permute(2,3,0,1).contiguous().view(-1, c).unsqueeze(1)
        cls_feat_norm = torch.norm(cls_feat_tmp, p=2, dim=2).squeeze()
        memo_neg_feat = memo_neg_feat.expand(h*w*n, c).unsqueeze(-1)
        memo_pos_feat = memo_pos_feat.expand(h*w,-1, -1).contiguous().view(h*w*n, -1).unsqueeze(-1)
        memo_neg_norm = torch.norm(memo_neg_feat,p=2, dim=1).squeeze()
        memo_pos_norm = torch.norm(memo_pos_feat,p=2, dim=1).squeeze()
        cos_dis_neg = torch.bmm(cls_feat_tmp, memo_neg_feat).squeeze()
        cos_dis_neg = cos_dis_neg/memo_neg_norm/cls_feat_norm
        cos_dis_pos = torch.bmm(cls_feat_tmp, memo_pos_feat).squeeze()
        cos_dis_pos = cos_dis_pos/memo_pos_norm/cls_feat_norm

        cos_dis_neg = cos_dis_neg.view(h,w,n).permute(2,0,1).unsqueeze(1)
        cos_dis_pos = cos_dis_pos.view(h,w,n).permute(2,0,1).unsqueeze(1)
        cos_dis = torch.cat((cos_dis_pos, cos_dis_neg), dim=1)
        cos_dis_norm = F.softmax(cos_dis, dim=1)
        cos_dis_abs = torch.pow(torch.abs(cos_dis_norm[:, :-1, ...] - cos_dis_norm[:, 1:, ...]), beta)
        bin_weight = torch.exp((cos_dis_abs-1.)*alpha)
        cls_mask = (cos_dis_pos > cos_dis_neg).float()

        return loc_map_cls, cls_mask, bin_weight

    def get_loss_sep(self,logits, gt_child_label, protype_h=None, protype_v=None, epoch=0, epoch_th=20):
        n,c, h,w  = logits.size()

        loss = 0
        if self.args.mce:
            ## 1. first calculate loss for each individual then reduction
            gt_child_label = gt_child_label.unsqueeze(-1).unsqueeze(-1)
            gt_child_label = gt_child_label.expand(n, h, w)
            loss += self.loss_cross_entropy(logits, gt_child_label.long())

            ## 2. firt reduction then calculate loss
            # logits = F.softmax(logits, dim=1)
            # logits = torch.mean(torch.mean(logits,dim=2), dim=2).clamp(min=1e-13)
            # loss += F.nll_loss(torch.log(logits), gt_child_label.long())

        return  loss


    def get_cls_maps(self):
        return F.relu(self.cls_map)
    def get_loc_maps(self):
        return torch.sigmoid(self.loc_map)

def normalize_feat(feat):
    n, fh, fw = feat.size()
    feat = feat.view(n, -1)
    min_val, _ = torch.min(feat, dim=-1, keepdim=True)
    max_val, _ = torch.max(feat, dim=-1, keepdim=True)
    norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
    norm_feat = norm_feat.view(n, fh, fw)

    return norm_feat

class memomy_module_v0(object):
    def __init__(self, n_classes, fg_th=0.6, bg_th=0.2, lr=0.5, lr_init=0.1, lr_epoch=5):
        self.n_classes = n_classes
        self._memo_kv = dict()
        for i in range(n_classes+1):
            self._memo_kv[i] = None

        self.lr = lr
        self.lr_init = lr_init
        self.lr_epoch = lr_epoch

        self.th_fg = fg_th
        self.th_bg = bg_th

    def update(self, feat_kv, epoch):
        if epoch >= self.lr_epoch:
            lr = self.lr
        else:
            lr = self.lr_init
        for k, v in feat_kv.items():
            if self._memo_kv[k] is None:
                self._memo_kv[k] = v
            else:
                delta_v = self._memo_kv[k] - v
                self._memo_kv[k] -= lr*delta_v

    def get_v(self, k):
        if isinstance(k, list):
            v= []
            for k_i in k:
                v.append(self._memo_kv[k_i].unsqueeze(0))
            v = torch.cat(v,dim=0)
        else:
            v = self._memo_kv[k].unsqueeze(0)
        return v

    def get_kv(self, feat, logits, target_loc_map, gt_labels):
        self.feat_cls_cur = feat
        n, _, lh, lw = target_loc_map.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = normalize_feat(var_logits)
        norm_var_logits = F.interpolate(norm_var_logits.unsqueeze(1), size=(lh, lw), mode='bilinear',
                                        align_corners=True)

        fg_cls = logits[torch.arange(n), gt_labels.long(), ...].clone()
        fg_cls = normalize_feat(fg_cls)
        norm_fg_cls = F.interpolate(fg_cls.unsqueeze(1), size=(lh, lw), mode='bilinear', align_corners=True)

        cls_mask = -1 * torch.ones_like(norm_var_logits)
        cls_mask[norm_var_logits < self.th_bg] = 0.
        cls_mask[norm_fg_cls > self.th_fg] = 1.

        memo_feat = dict()
        gt_labels_tmp = gt_labels.long().cpu().numpy()
        for i in range(n):
            k = gt_labels_tmp[i]
            if k not in memo_feat:
                memo_feat[k] = feat[i,:,cls_mask[i,0]>0]
            else:
                memo_feat[k] = torch.cat((memo_feat[k],feat[i, :, cls_mask[i, 0] > 0]), dim=1)
            if 200 not in memo_feat:
                memo_feat[200] = feat[i, :, cls_mask[i, 0] == 0]
            else:
                memo_feat[200] = torch.cat((memo_feat[200], feat[i, :, cls_mask[i, 0] == 0]), dim=1)

        for k, v in memo_feat.items():
            memo_feat[k] = torch.mean(v, dim=1)

        return  memo_feat


class cls_fea_hv(nn.Module):
    def __init__(self,f_in, f_out,):
        super(cls_fea_hv,self).__init__()
        self.fc_h = nn.Linear(f_in, f_out)
        self.fc_v = nn.Linear(f_in, f_out)
    def forward(self, x):
        x = torch.mean(x, dim=2)
        x_h = torch.mean(x, dim=2)
        x_v = torch.mean(x, dim=3)
        x_h =self.fc_h(x_h)
        x_v = self.fc_v(x_v)
        return x_h, x_v

def make_layers(cfg, dilation=None, batch_norm=False, instance_norm=False, inl=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'L':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif instance_norm and v <256 and v>64:
                layers += [conv2d, nn.InstanceNorm2d(v, affine=inl), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    # 'D_deeplab': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}

dilation = {
    # 'D_deeplab': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 2, 2, 2, 'N'],
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}

cnvs= {'O': (10,7,7), 'OI':(12,7,7)}

def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """

    layers = make_layers(cfg['O'], dilation=dilation['D1'])
    cnv = np.cumsum(cnvs['OI']) if kwargs['args'].IN or kwargs['args'].INL else np.cumsum(cnvs['O'])
    model = VGG(layers, cnvs=cnv, **kwargs)
    if pretrained:
        pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.weight'.format(i + 10), 'conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 10), 'conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 17), 'conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 17), 'conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 24), 'conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 24), 'conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)


        model_dict = model.state_dict()
        pretrained_file = os.path.join(kwargs['args'].pretrained_model_dir, kwargs['args'].pretrained_model)
        if os.path.isfile(pretrained_file):
            pretrained_dict = torch.load(pretrained_file)
            print('load pretrained model from {}'.format(pretrained_file))
        else:
            pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
            print('load pretrained model from {}'.format(model_urls['vgg16']))
        # 0. replace the key
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}
        # *. show the loading information
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from vgg16'.format(k))
        print(' ')
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

if __name__ == '__main__':
    model(True)
