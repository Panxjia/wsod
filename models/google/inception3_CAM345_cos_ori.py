import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from torch.autograd import Variable
import os
import cv2
import numpy as np
from ..carafe.carafe import CARAFEPack

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
        self.num_maps = 1 if args.NoDDA else int(args.num_maps)
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


        if self.args.carafe:
            # carafe
            self.classifier6 = nn.Sequential(
                nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                CARAFEPack(1024,1,up_kernel=3,compressed_channels=256),
                nn.ReLU(True),
                nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0)
            )
            # carafe1
            # self.classifier6 = nn.Sequential(
            #     CARAFEPack(768, 1, up_kernel=3, compressed_channels=256),
            #     nn.ReLU(True),
            #     nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0)
            # )
        elif self.args.carafe_cls:
            # cls
            # self.classifier6 = nn.Sequential(
            #     nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            #     nn.ReLU(True),
            #     CARAFEPack(1024, 1, up_kernel=3, compressed_channels=256, normalized=False),
            #     nn.ReLU(True),
            #     nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0)
            # )
            #cls_1
            self.classifier6 = nn.Sequential(
                nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0),
                nn.ReLU(True),
                CARAFEPack(self.sup_child_classes * self.num_maps, 1, up_kernel=3, compressed_channels=200, compress=False, normalized=False)
            )
        elif self.args.non_local:
            self.final_fea = nn.Sequential(
                nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True))
            self.non_local_phi = nn.Conv2d(1024,512,kernel_size=1)
            self.non_local_theta = nn.Conv2d(1024,512,kernel_size=1)
            self.non_local_g = nn.Conv2d(1024,512,kernel_size=1)
            self.channel_exp = nn.Conv2d(512,1024,kernel_size=1)
            self.classifier = nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0)
        else:
            self.classifier6 = nn.Sequential(
                nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0)
            )
            # self.classifier6 = nn.Sequential(
            #     nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(1024, self.sup_child_classes * self.num_maps, kernel_size=1, padding=0),
            #     nn.ReLU(True),
            #     nn.Conv2d(self.sup_child_classes * self.num_maps, self.sup_child_classes * self.num_maps, kernel_size=3, padding=1)
            # )
        self.feature_loc = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
        )
        self.localizer = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(1024, 1, kernel_size=1, padding=0)
        )


        # ================================ loss ===================================
        # self.loss_cross_entropy = nn.CrossEntropyLoss()
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
        if self.args.non_local:
            cam = self.non_local(features, kernel=self.args.non_local_kernel)
        else:
            cam = self.classifier6(features)
        fea_loc = self.feature_loc(features)
        self.child_map = cam
        self.fea_loc = fea_loc
        self.bin_map = self.localizer(fea_loc)
        self.root_map = self.root_map.view(batch_size, self.sup_root_classes, self.num_maps, 25, 25)
        self.parent_map = self.parent_map.view(batch_size, self.sup_parent_classes, self.num_maps, 25, 25)
        self.child_map = self.child_map.view(batch_size, self.sup_child_classes, self.num_maps, 12, 12)


        return self.root_map, self.parent_map, self.child_map
        # return root_logits, parent_logits, child_logits

    ### nonlocal3
    def non_local(self, features, kernel=3):
        feat = self.final_fea(features)
        n, c, h, w = feat.size()

        f_phi = self.non_local_phi(feat)
        f_theta = self.non_local_theta(feat)
        f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, 512)
        f_theta = f_theta.contiguous().view(n, 512, -1)
        non_local_cos = torch.matmul(f_phi, f_theta)
        feat_neighbor_mat = torch.zeros_like(non_local_cos)
        if kernel >0:
            feat_neighbor_loc = self.neighbor_area(h, w, kernel=kernel)
            ind = torch.arange(int(h*w)).view(-1,1)
            ind = ind.repeat(1,kernel*kernel)
            feat_neighbor_mat[:,ind,feat_neighbor_loc] = 1
            ## pairwise function
            # 1. -- guassian embedding--
            if self.args.non_local_pf == 0:
                non_local_cos[feat_neighbor_mat < 1] = -1e15
                non_local_cos = F.softmax(non_local_cos, dim=2)

            # 2. -- dot production--
            elif self.args.non_local_pf == 1:
                non_local_cos[feat_neighbor_mat < 1] = 0
                norm_n = feat_neighbor_mat.sum(dim=-1, keepdim=True)
                non_local_cos = non_local_cos / norm_n
            else:
                print('Wrong value of non local pairwise function.')
        else:
            ## pairwise function
            # 1. -- guassian embedding--
            if self.args.non_local_pf == 0:
                non_local_cos = F.softmax(non_local_cos, dim=2)

            # 2. -- dot production--
            elif self.args.non_local_pf == 1:
                norm_n = torch.ones_like(non_local_cos)*(h*w)
                non_local_cos = non_local_cos/norm_n
            else:
                print('Wrong value of non local pairwise function.')

        final_feat = feat.permute(0, 2, 3, 1).contiguous().view(n, -1, c)
        final_feat = torch.matmul(non_local_cos, final_feat)
        final_feat = final_feat.contiguous().view(n, h, w, c).permute(0, 3, 1, 2)
        if self. args.non_local_res:
            cam = self.classifier(final_feat+feat)
        else:
            cam = self.classifier(final_feat)
        return cam



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

    def get_loss(self, logits, gt_root_label, gt_parent_label, gt_child_label, child_h = None,
                 child_v=None, parent_h=None, parent_v=None, root_h=None,root_v=None, epoch=0, epoch_th=125):
        root_map, parent_map, child_map = logits

        if epoch > epoch_th:
            root_logits = self.get_logits(root_map, gt_root_label, self.args.cls_th, self.args.bak_fac)
            parent_logits = self.get_logits(parent_map, gt_parent_label, self.args.cls_th, self.args.bak_fac)
            child_logits = self.get_logits(child_map, gt_child_label ,self.args.cls_th, self.args.bak_fac)
        else:
            root_logits = torch.mean(torch.mean(torch.mean(root_map, dim=2), dim=2), dim=2)
            parent_logits = torch.mean(torch.mean(torch.mean(parent_map, dim=2), dim=2),dim=2)
            child_logits = torch.mean(torch.mean(torch.mean(child_map, dim=2), dim=2),dim=2)
        loss_cos =None
        if not self.args.NoDDA:
            maps = self.get_gt_map(gt_root_label, gt_parent_label, gt_child_label)
            loss_cos, random_seed = self.calculate_cosineloss(maps)
        if self.args.bce:
            n = root_logits.size(0)
            gt_root = torch.zeros_like(root_logits)
            gt_root[torch.arange(n), gt_root_label.long()] = 1

            gt_parent = torch.zeros_like(parent_logits)
            gt_parent[torch.arange(n), gt_parent_label.long()] = 1

            gt_child = torch.zeros_like(child_logits)
            gt_child[torch.arange(n), gt_child_label.long()] = 1
            if self.args.weight_bce:
                gt_root_w = torch.ones_like(root_logits)*(1-self.args.bce_pos_weight)/(root_logits.size(1)-1)
                gt_root_w[torch.arange(n), gt_root_label.long()] = self.args.bce_pos_weight

                gt_parent_w = torch.ones_like(parent_logits) * (1 - self.args.bce_pos_weight)/(parent_logits.size(1)-1)
                gt_parent_w[torch.arange(n), gt_parent_label.long()] = self.args.bce_pos_weight

                gt_child_w = torch.ones_like(child_logits) * (1 - self.args.bce_pos_weight)/(child_logits.size(1)-1)
                gt_child_w[torch.arange(n), gt_child_label.long()] = self.args.bce_pos_weight

                root_loss_cls = self.loss_bce(root_logits, gt_root, reduction='none') * gt_root_w
                parent_loss_cls = self.loss_bce(parent_logits, gt_parent, reduction='none') * gt_parent_w
                child_loss_cls = self.loss_bce(child_logits, gt_child, reduction='none') * gt_child_w
                root_loss_cls = root_loss_cls.sum(dim=1).mean()
                parent_loss_cls = parent_loss_cls.sum(dim=1).mean()
                child_loss_cls = child_loss_cls.sum(dim=1).mean()
            else:
                root_loss_cls = self.loss_bce(root_logits, gt_root)
                parent_loss_cls = self.loss_bce(parent_logits, gt_parent)
                child_loss_cls = self.loss_bce(child_logits, gt_child)
        else:
            root_loss_cls = self.loss_cross_entropy(root_logits, gt_root_label.long())
            parent_loss_cls = self.loss_cross_entropy(parent_logits, gt_parent_label.long())
            child_loss_cls = self.loss_cross_entropy(child_logits, gt_child_label.long())
        if self.args.NoHDA:
            loss_val = 0.5 * child_loss_cls
            if not self.args.NoDDA:
                loss_val += self.cos_alpha * loss_cos
            if self.args.bin_cls:
                mask = self.get_cls_mask(child_map, gt_child_label, self.args.cls_th_h, self.args.cls_th_l)
                loss_sim = self.get_sim_loss(self.fea_loc,mask)
                loss_bin = self.get_bin_loss(self.bin_map, mask)
                loss_loc= 0.5*loss_bin + self.args.sim_alpha * loss_sim
                loss_val = loss_val + loss_loc
        else:
            loss_val = 0.5 * root_loss_cls + 0.5 * parent_loss_cls + 0.5 * child_loss_cls + self.cos_alpha * loss_cos
        loss_sim = torch.zeros_like(loss_val)

        return loss_val, root_loss_cls, parent_loss_cls, child_loss_cls, \
               loss_cos, loss_sim, root_logits, parent_logits, child_logits

    def get_cls_mask(self, cls_map, gt_label, thr_h, thr_l):
        cls_map_c = torch.mean(F.relu(cls_map), dim=2)
        b, c, h, w = cls_map_c.size()
        cls_map_c = cls_map_c.contiguous().view(b, c, -1).detach()
        cls_map_c = cls_map_c[torch.arange(b), gt_label.long()]
        min = torch.min(cls_map_c, dim=1, keepdim=True)[0]
        max = torch.max(cls_map_c, dim=1, keepdim=True)[0]
        norma_cls_map = (cls_map_c - min) / (max - min + 1e-10)
        norma_cls_map = norma_cls_map.contiguous().view(b, h, w)
        bin_mask = torch.zeros_like(norma_cls_map)
        bin_mask[norma_cls_map > thr_l] = -1
        bin_mask[norma_cls_map > thr_h] = 1
        return bin_mask.unsqueeze(1)

    def get_bin_loss(self, pre_bin_map, cls_mask):
        bin_weight = torch.ones_like(cls_mask)
        bin_weight[cls_mask<0] = 0
        cls_mask[cls_mask<0] = 0
        bin_loss = self.loss_bce(pre_bin_map,cls_mask,reduction='none')
        bin_loss  = bin_loss * bin_weight
        return torch.sum(bin_loss)/torch.sum(bin_weight)

    def get_sim_loss(self, fea_loc, cls_mask):
        n, c, _,_ = fea_loc.size()
        n_sample = torch.sum(cls_mask==-1)
        cls_mask = cls_mask.expand(fea_loc.size())
        loss_sim = 0
        for i in range(n):
            fea_mean = torch.mean((fea_loc[i,...][cls_mask[i,...]==1]).view(c,-1), dim=-1, keepdim=True)
            fea_mid = (fea_loc[i][cls_mask[i]==-1]).view(c,-1)
            fea_sim = torch.sum(fea_mid * fea_mean, dim=0)
            fea_len_mid = torch.norm(fea_mid, 2, dim=0)
            fea_len_mean = torch.norm(fea_mean, 2, dim=0)
            loss_sim += torch.sum(1.0 - fea_sim / fea_len_mean / fea_len_mid)
        loss_sim /= n_sample
        return loss_sim
    def get_logits(self, cls_map, gt_label,cls_th, bak_factor=0.2):
        cls_map_c = torch.mean(F.relu(cls_map), dim=2)
        cls_map = torch.mean(cls_map, dim=2)
        b, c, h, w = cls_map.size()
        cls_map = cls_map.contiguous().view(b,c,-1)
        cls_map_c = cls_map_c.contiguous().view(b,c,-1).detach()
        cls_map_c = cls_map_c[torch.arange(b), gt_label.long()]
        min = torch.min(cls_map_c, dim=1, keepdim=True)[0]
        max = torch.max(cls_map_c, dim=1, keepdim=True)[0]
        norma_cls_map = (cls_map_c - min)/(max - min +1e-10)
        mask = norma_cls_map < cls_th
        cls_map_new = cls_map.clone()
        for i_batch in range(b):
            cls_map_new[i_batch,:,mask[i_batch]] = cls_map[i_batch,:,mask[i_batch]] * bak_factor
            # cls_map[i_batch,:,mask[i_batch]] *= bak_factor
            # logits.append(torch.mean(cls_map[i_batch, :,mask[i_batch]], dim=1).unsqueeze(0))
        # return torch.cat(logits, dim=0)
        return torch.mean(cls_map_new, dim=2)

    def get_logits_from_score(self, cls_map, gt_label,cls_th):
        b, c, h, w = cls_map.size()
        cls_score = F.softmax(cls_map, dim=2).detach()
        cls_score = cls_score.contiguous().view(b,c,-1)
        cls_map = cls_map.contiguous().view(b,c,-1)
        cls_score_c = cls_score[torch.arange(b), gt_label.long()]
        mask = cls_score_c > cls_th
        logits = []
        for i_batch in range(b):
            logits.append(torch.mean(cls_map[i_batch, :,mask[i_batch]], dim=1).unsqueeze(0))
        return torch.cat(logits, dim=0)


    def get_child_maps(self):
        return torch.mean(F.relu(self.child_map), dim=2)

    def get_parent_maps(self):
        return torch.mean(F.relu(self.parent_map), dim=2)

    def get_root_maps(self):
        return torch.mean(F.relu(self.root_map), dim=2)
    def get_bin_map(self):
        return torch.sigmoid(self.bin_map)

    def cls_fea_hv(self,gt_label, feature_map, width, height, att_th=0.15):
        n = gt_label.size(0)
        cls_map = feature_map[torch.arange(n),gt_label.long(),...]
        min = cls_map.min(dim=-1).min(dim=-1)
        max = cls_map.max(dim=-1).max(dim=-1)
        norm_cls_map = (cls_map-min)/(max - min +1e-10)
        cls_fea_h =[]
        cls_fea_w =[]
        for i in range(n):
            pass

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

