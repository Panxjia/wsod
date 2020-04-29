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
        self.conv5 = nn.Sequential(*features[cnvs[2]:])
        self.num_classes = num_classes
        self.args = args

        self.cls5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0),
        )
        self.cls4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0),
        )
        self.cls3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.num_classes, kernel_size=1, padding=0),
        )

        # self.cls_4 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        # self.cls_5 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)

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

        if self.args.loc_branch:
            self.loc = nn.Sequential(
                nn.Conv2d(128, 512, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
                nn.Conv2d(512, 1, kernel_size=1, padding=0)
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



    def forward(self, x):
        x = self.conv1_2(x)
        feat_3 = self.conv3(x)
        self.feat_3 = feat_3
        feat_4 = self.conv4(feat_3)
        self.feat_4 = feat_4
        feat_5 = self.conv5(feat_4)
        self.feat_5 = feat_5
        self.construct_fpn()
        cls_map_3 = self.cls3(self.f3_2)
        cls_map_4 = self.cls4(self.f4_2)
        cls_map_5 = self.cls5(self.f5_2)

        if self.args.loc_branch:
            if self.args.com_feat:
                n,c,h,w = feat_3.size()
                feat_45 = self.conv45(feat_4)
                feat_45 = F.interpolate(feat_45,size=(h,w),mode='bilinear', align_corners=True)
                feat_55 = self.conv55(feat_5)
                feat_55 = F.interpolate(feat_55,size=(h,w),mode='bilinear', align_corners=True)
                feat_35 = self.conv35(feat_3)
                merge_feat = self.merge(feat_45+ feat_55+ feat_35)
                loc_map = self.loc(merge_feat)
                self.loc_map = loc_map
            else:
                loc_map = self.loc(feat_5)
                self.loc_map = loc_map

        return cls_map_3, cls_map_4, cls_map_5

    def construct_fpn(self):
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

            f5_0, f4_0, f3_0 = self.feat_5, self.feat_4, self.feat_3
            f5_0_41 = self.fpn_f5_0_41(f5_0)
            self.fpn_f5_0_41_up = F.interpolate(f5_0_41, scale_factor=2, mode='bilinear', align_corners=True)

            f4_1 = self.fpn_mix4_1(self.fpn_f5_0_41_up + self.fpn_f4_0_1(f4_0))
            self.f4_1_up = F.interpolate(f4_1, scale_factor=2, mode='bilinear', align_corners=True)

            f3_2 = self.fpn_mix3_2(self.fpn_f3_0_2(f3_0) + self.f4_1_up)
            f4_2 = self.fpn_mix4_2(self.fpn_f4_0_2(f4_0) + self.fpn_f4_1_2(f4_1) + self.fpn_f3_2_down(f3_2) )
            f5_2 = self.fpn_mix5_2(self.fpn_f5_0_2(f5_0)+self.fpn_f4_2_down(f4_2))

            self.f3_2 = f3_2
            self.f4_2 = f4_2
            self.f5_2 = f5_2
        if self.args.fpn:
            f5_0, f4_0, f3_0 = self.feat_5, self.feat_4, self.feat_3
            lateral_5_conv = self.fpn_lat_5(f5_0)
            self.f5_2 = self.fpn_out_5(lateral_5_conv) + f5_0
            f5_up = F.interpolate(lateral_5_conv, scale_factor=2, mode='bilinear', align_corners=True)
            lateral_4_conv = self.fpn_lat_4(f4_0) + f5_up
            self.f4_2 = self.fpn_out_4(lateral_4_conv) + f4_0
            f4_up = F.interpolate(lateral_4_conv, scale_factor=2, mode='bilinear', align_corners=True)
            self.f3_2 = self.fpn_out_3(self.fpn_lat_3(f3_0) + f4_up) + f3_0


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
    def norm_atten_map(self, map):
        min_val, _ = torch.min(torch.min(map, dim=-1, keepdim=True)[0], dim=-1, keepdim=True)
        max_val, _ = torch.max(torch.max(map, dim=-1, keepdim=True)[0], dim=-1, keepdim=True)
        norm_map = (map - min_val) / (max_val - min_val + 1e-15)
        return norm_map

    def get_loss(self, logits, gt_child_label, protype_h=None, protype_v=None, epoch=0, loc_start=10, erase_start=10):

        logits_3, logits_4, logits_5 = logits
        if self.args.erase and epoch >= erase_start:
            n, c = logits_3.size()[:2]
            atten_map_3 = logits_3[torch.arange(n), gt_child_label.long(), ...]
            atten_map_4 = logits_4[torch.arange(n), gt_child_label.long(), ...]

            norm_atten_map_3 = self.norm_atten_map(atten_map_3.clone())
            norm_atten_map_4 = self.norm_atten_map(atten_map_4.clone())

            norm_atten_mask_3 = norm_atten_map_3 < self.args.erase_th
            norm_atten_mask_3 = norm_atten_mask_3.float().unsqueeze(1).repeat(1,c,1,1)
            norm_atten_mask_4 = norm_atten_map_4 < self.args.erase_th
            norm_atten_mask_4 = norm_atten_mask_4.float().unsqueeze(1).repeat(1,c,1,1)

            norm_atten_mask_3 = F.max_pool2d(norm_atten_mask_3, kernel_size=2, stride=2)
            norm_atten_mask_4 = F.max_pool2d(norm_atten_mask_4, kernel_size=2, stride=2)

            logits_4 = logits_4 * norm_atten_mask_3.detach()
            logits_5 = logits_5 * norm_atten_mask_4.detach()

        cls_logits_3 = torch.mean(torch.mean(logits_3, dim=2), dim=2)
        cls_logits_4 = torch.mean(torch.mean(logits_4, dim=2), dim=2)
        cls_logits_5 = torch.mean(torch.mean(logits_5, dim=2), dim=2)

        loss_3 = self.loss_cross_entropy(cls_logits_3, gt_child_label.long())
        loss_4 = self.loss_cross_entropy(cls_logits_4, gt_child_label.long())
        loss_5 = self.loss_cross_entropy(cls_logits_5, gt_child_label.long())
        loss = self.args.loss_w_3 * loss_3 + self.args.loss_w_4 * loss_4 + self.args.loss_w_5 * loss_5

        if self.args.loc_branch and epoch >= loc_start:
            loc_loss = self.get_loc_loss(logits, gt_child_label, self.args.th_bg, self.args.th_fg)
            loss += loc_loss
        else:
            loc_loss = torch.zeros_like(loss)
        return loss, loss_3, loss_4, loss_5, loc_loss

    def get_loc_loss(self, logits, gt_child_label, th_bg=0.3, th_fg=0.5):
        n, c, lh, lw = self.loc_map.size()
        if self.args.avg_bin:
            loc_map_cls = F.avg_pool2d(self.loc_map, kernel_size=self.args.avg_size, stride=self.args.avg_stride)
            # loc_map_cls = F.adaptive_avg_pool2d(self.loc_map, output_size=1)
        else:
            loc_map_cls = self.loc_map

        _, _, h, w = logits.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = self.normalize_feat(var_logits)
        norm_var_logits = F.interpolate(norm_var_logits.unsqueeze(1),size=(lh,lw), mode='bilinear', align_corners=True)

        fg_cls = cls_logits[torch.arange(n), gt_child_label.long(), ...].clone()
        fg_cls = self.normalize_feat(fg_cls)
        norm_fg_cls = F.interpolate(fg_cls.unsqueeze(1),size=(lh,lw), mode='bilinear', align_corners=True)
        # norm_fg_cls = norm_fg_cls.squeeze()

        norm_cls = (norm_var_logits + norm_fg_cls) * 0.5
        cls_mask = -1 * torch.ones_like(norm_var_logits)
        cls_mask[norm_cls < th_bg] = 0.
        cls_mask[norm_cls > th_fg] = 1.

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

        # cls_mask = -1 * torch.ones_like(norm_var_logits)
        # cls_mask[norm_var_logits < th_bg ] = 0.
        # cls_mask[norm_fg_cls > th_fg] = 1.
        # # cls_mask = cls_mask.unsqueeze(1)
        # if self.args.avg_bin:
        #     cls_mask = F.max_pool2d(cls_mask, kernel_size=self.args.avg_size, stride=self.args.avg_stride)
        #     # cls_mask = F.adaptive_max_pool2d(cls_mask,output_size=1)
        # if self.args.adap_w:
        #     # bin_weight = torch.exp(self.args.adap_w_gama*torch.abs(norm_var_logits - norm_fg_cls))
        #     loc_logits = torch.sigmoid(loc_map_cls)
        #     bin_weight_pos = cls_mask * (torch.pow((1-loc_logits),self.args.adap_w_gama))
        #     bin_weigt_neg = (1-cls_mask)* (torch.pow(loc_logits,self.args.adap_w_gama))
        #     bin_weight = bin_weight_pos + bin_weigt_neg
        #     bin_weight[cls_mask<0] =0.
        #
        # else:
        #     bin_weight = torch.ones_like(cls_mask)
        #     bin_weight[cls_mask<0] =0.

        # loc_map_cls = F.adaptive_avg_pool2d(self.loc_map, output_size=1)
        # cls_mask = torch.ones_like(loc_map_cls)
        # bin_loss = self.loss_bce(loc_map_cls, cls_mask)
        # return bin_loss

        bin_loss = self.loss_bce(loc_map_cls, cls_mask, reduction='none')
        bin_loss = bin_loss * bin_weight
        return torch.sum(bin_loss) / torch.sum(bin_weight)

        # loss = self.loss_cross_entropy(self.loc_map, gt_child_label_cp.long(), reduction='none')
        # loss = torch.sum(loss * pos_sam)
        # return loss/float(torch.sum(pos_sam))

    def memo_prot(self):
        pass

    def normalize_feat(self,feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)

        return norm_feat

    def get_loss_sep(self,logits, gt_child_label, protype_h=None, protype_v=None, epoch=0, epoch_th=20):
        n,c, h,w  = logits.size()

        loss = 0
        if self.args.mce:
            ## 1. first calculate loss for each individual then reduction
            # gt_child_label = gt_child_label.unsqueeze(-1).unsqueeze(-1)
            # gt_child_label = gt_child_label.expand(n, h, w)
            # loss += self.loss_cross_entropy(logits, gt_child_label.long())
            # logits = torch.mean(torch.mean(logits, dim=2), dim=2)

            ## 2. firt reduction then calculate loss
            logits = F.softmax(logits, dim=1)
            logits = torch.mean(torch.mean(logits,dim=2), dim=2).clamp(min=1e-13)
            loss += F.nll_loss(torch.log(logits), gt_child_label.long())

        return  loss, logits


    def get_cls_maps(self):
        return F.relu(self.cls_map)
    def get_loc_maps(self):
        return torch.sigmoid(self.loc_map)




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
