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
        self.cos_alpha = args.cos_alpha
        self.num_maps = int(args.num_maps)
        self.args = args
        self.root_num_classes = 11
        self.parent_num_classes = 37
        self.child_num_classes = num_classes

        self.sup_root_classes = 11+args.sup if args.bbce else 11
        self.sup_parent_classes = 37+args.sup if args.bbce else 37
        self.sup_child_classes = num_classes+args.sup if args.bbce else num_classes
        # added layer
        if args.RGAP:
            self.fc3_1 = residual_gap_block(256,512,3,1,1)
            self.fc3_2 = residual_gap_block(512,512,3,1,1)
        else:
            self.fc3_1 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
            )
            self.fc3_2 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
            )
        self.cls3_bf = nn.Conv2d(512, self.sup_root_classes*self.num_maps, kernel_size=1, padding=0)  #
        self.cls3 = nn.Conv2d(512, self.sup_root_classes*self.num_maps, kernel_size=1, padding=0)  #
        if args.lb:
            self.fc3_1_lb = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
            )
            self.fc3_2_lb = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
            )
            self.cls3_ce = nn.Conv2d(512, self.root_num_classes*self.num_maps, kernel_size=1, padding=0)
        if args.RGAP:
            self.fc4_1 = residual_gap_block(512,1024,3,1,1)
            self.fc4_2 = residual_gap_block(1024,1024,3,1,1)
        else:
            self.fc4_1 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
            )
            self.fc4_2 = nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
            )
        self.cls4 = nn.Conv2d(1024, self.sup_parent_classes*self.num_maps, kernel_size=1, padding=0)  #
        if args.lb:
            self.fc4_1_lb = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
            )
            self.fc4_2_lb = nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
            )
            self.cls4_ce = nn.Conv2d(1024, self.parent_num_classes*self.num_maps, kernel_size=1, padding=0)  #
        if args.RGAP:
            self.fc5_1 = residual_gap_block(512,1024,3,1,1)
            self.fc5_2 = residual_gap_block(1024,1024,3,1,1)
        else:
            self.fc5_1 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
            )
            self.fc5_2 = nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
            )
        self.cls5 = nn.Conv2d(1024, self.sup_child_classes*self.num_maps, kernel_size=1, padding=0)  #
        if args.lb:
            self.fc5_1_lb = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
                nn.ReLU(True),
            )
            self.fc5_2_lb = nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
                nn.ReLU(True),
            )
            self.cls5_ce = nn.Conv2d(1024, self.child_num_classes*self.num_maps, kernel_size=1, padding=0)  #
        if args.sc:
            self.cls5_sc = cls_fea_hv(14,7)
            self.cls4_sc = cls_fea_hv(14,7)
            self.cls3_sc = cls_fea_hv(28,7)
        self._initialize_weights()

        # loss function
        if args.trunc_loss:
            self.loss_cross_entropy = trunctable_cross_entropy(threshold=args.loss_trunc_th)
            self.loss_bce = trunctable_cross_entropy(threshold=args.loss_trunc_th, softmax=False)
        else:
            self.loss_cross_entropy = F.cross_entropy
            self.loss_bce = F.binary_cross_entropy_with_logits
        if args.eval_gcam:
            self.guided_grad = None
            self.last_layer_grad_out = {}
            self.last_layer_feat = {}
            self.act_maps = {}
            self.register_hooks()

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
    def register_hooks(self):
        def first_layer_hook_bw(module, grad_in, grad_out):
            self.guided_grad = grad_in[0]

        def last_layer_hook_bk( name):
            def hook_bw(module, grad_in, grad_out):
                self.last_layer_grad_out[name] = torch.clamp(grad_out[0], min=0.0)
            return hook_bw

        def last_layer_hook_fk(name):
            def hook_fw(module, input, output):
                self.last_layer_feat[name] = F.relu(output)
            return hook_fw
        def forward_hook_fn(name):
            def hook_fw(module, input, output):
                self.act_maps[name] = output
            return hook_fw
        def backward_hook_fn( name):
            def hook_bw(module, grad_in, grad_out):
                grad = self.act_maps[name].clone()
                grad[grad > 0] = 1.
                positive_grad_out = torch.clamp(grad_out[0], min=0.0)
                new_grad_in = positive_grad_out * grad
                return (new_grad_in,)
            return hook_bw

        for i, (n, m) in enumerate(self.named_modules()):
            if isinstance(m, nn.ReLU):
                m.register_forward_hook(forward_hook_fn(n))
                m.register_backward_hook(backward_hook_fn(n))
        self.cls3.register_forward_hook(last_layer_hook_fk(name='cls3'))
        self.cls3.register_backward_hook(last_layer_hook_bk(name='cls3'))
        self.cls4.register_forward_hook(last_layer_hook_fk(name='cls4'))
        self.cls4.register_backward_hook(last_layer_hook_bk(name='cls4'))
        self.cls5.register_forward_hook(last_layer_hook_fk( name='cls5'))
        self.cls5.register_backward_hook(last_layer_hook_bk(name='cls5'))
        self.conv1_2[0].register_backward_hook(first_layer_hook_bw)
        # for m in self.conv1_2[:1].modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.register_backward_hook(first_layer_hook_fn)


    def forward(self, x):
        # ======================================================================
        self.guided_grad = None
        self.last_layer_grad_out = {}
        self.last_layer_feat = {}
        self.act_maps = {}

        x = self.conv1_2(x)
        x = self.conv3(x)
        batch_size = x.size(0)
        rootResult = self.fc3_1(x)
        rootResult = self.fc3_2(rootResult)
        rootResult_multimaps = self.cls3(rootResult).view(batch_size, self.sup_root_classes, self.num_maps, 28, 28)
        # rootResult = torch.sum(rootResult_multimaps, 2).view(batch_size, self.sup_root_classes, 28, 28) / self.num_maps
        # root_logits = torch.mean(torch.mean(rootResult, dim=2), dim=2)
        root_logits = rootResult_multimaps
        if self.args.lb:
            rootResult_ce = self.fc3_1_lb(x)
            rootResult_ce = self.fc3_2_lb(rootResult_ce)
            rootResult_multimaps_ce = self.cls3_ce(rootResult_ce).view(batch_size, self.root_num_classes, self.num_maps, 28, 28)
            # rootResult_ce = torch.sum(rootResult_multimaps_ce, 2).view(batch_size, self.root_num_classes, 28, 28) / self.num_maps
            # root_logits_ce = torch.mean(torch.mean(rootResult_ce, dim=2), dim=2)
            root_logits_ce = rootResult_multimaps_ce


        # # ======================================================================
        # =============================== Result root ==============================

        x = self.conv4(x)

        parentResult = self.fc4_1(x)
        parentResult = self.fc4_2(parentResult)
        parentResult_multimaps = self.cls4(parentResult).view(batch_size, self.sup_parent_classes, self.num_maps, 14, 14)
        # parentResult = torch.sum(parentResult_multimaps, 2).view(batch_size, self.sup_parent_classes, 14,
        #                                                          14) / self.num_maps
        # parent_logits = torch.mean(torch.mean(parentResult, dim=2), dim=2)
        parent_logits = parentResult_multimaps
        if self.args.lb:
            parentResult_ce = self.fc4_1_lb(x)
            parentResult_ce = self.fc4_2_lb(parentResult_ce)
            parentResult_multimaps_ce = self.cls4_ce(parentResult_ce).view(batch_size, self.parent_num_classes, self.num_maps, 14,
                                                                 14)
            # parentResult_ce = torch.sum(parentResult_multimaps_ce, 2).view(batch_size, self.parent_num_classes, 14,
            #                                                                14) / self.num_maps
            # parent_logits_ce = torch.mean(torch.mean(parentResult_ce, dim=2), dim=2)
            parent_logits_ce = parentResult_multimaps_ce


        # # ======================================================================
        # =============================== Result parent ==============================

        x = self.conv5(x)

        childResult = self.fc5_1(x)
        childResult = self.fc5_2(childResult)
        childResult_multimaps = self.cls5(childResult).view(batch_size, self.sup_child_classes, self.num_maps, 14, 14)
        # childResult = torch.sum(childResult_multimaps, 2).view(batch_size, self.sup_child_classes, 14,
        #                                                        14) / self.num_maps
        #
        # # child_logits = torch.mean(torch.mean(childResult, dim=2), dim=2)
        # child_logits = torch.mean(torch.mean(childResult, dim=2), dim=2)
        child_logits = childResult_multimaps
        if self.args.sc:
            self.child_cls_fea = self.cls5_sc(childResult_multimaps)
            self.parent_cls_fea = self.cls4_sc(parentResult_multimaps)
            self.root_cls_fea = self.cls3_sc(rootResult_multimaps)
        if self.args.lb:
            childResult_ce = self.fc5_1_lb(x)
            childResult_ce = self.fc5_2_lb(childResult_ce)
            childResult_multimaps_ce = self.cls5_ce(childResult_ce).view(batch_size, self.child_num_classes, self.num_maps, 14, 14)
            # childResult_ce = torch.sum(childResult_multimaps_ce, 2).view(batch_size, self.child_num_classes, 14,
            #                                                              14) / self.num_maps
            # child_logits_ce = torch.mean(torch.mean(childResult_ce, dim=2), dim=2)
            child_logits_ce = childResult_multimaps_ce

        # ======================================================================
        # =============================== Result child ===============================


        self.child_map = childResult_multimaps
        self.parent_map = parentResult_multimaps
        self.root_map = rootResult_multimaps
        if self.args.lb:
            self.child_map_ce = childResult_multimaps_ce
            self.parent_map_ce = parentResult_multimaps_ce
            self.root_map_ce = rootResult_multimaps_ce

        if self.args.lb:
            return root_logits, parent_logits, child_logits, root_logits_ce, parent_logits_ce, child_logits_ce
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

        # maps_max = maps.max(dim=2)[0].expand(maps.shape[-1], batch_size, channel_num).permute(1, 2, 0)
        # maps = maps/maps_max
        X1 = maps.unsqueeze(1)
        X2 = maps.unsqueeze(2)
        dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
        # print(dot12)
        dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))
        tri_tensor = ((torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1]*channel_num))).expand(batch_size, channel_num, channel_num)).cuda()
        dist_num = abs((tri_tensor*dist).sum(1).sum(1)).sum()/(batch_size*channel_num*(channel_num-1)/2)

        return dist_num, random_seed


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


    def get_gt_map(self, gt_root_label, gt_parent_label, gt_child_label):
        batch_size = self.child_map.size(0)
        child_map = self.child_map.reshape(batch_size*self.sup_child_classes, self.num_maps, 14, 14).clone()
        child_map = child_map[[gt_child_label[i].long()+(i*self.sup_child_classes) for i in range(batch_size)], :, :, :]
        child_map = F.interpolate(child_map.reshape(batch_size, self.num_maps, 14, 14),size=(28, 28), mode='bilinear', align_corners=True)

        if self.args.NoHDA:
            return child_map

        parent_map = self.parent_map.reshape(batch_size * self.sup_parent_classes, self.num_maps, 14, 14).clone()
        parent_map = parent_map[[gt_parent_label[i].long() + (i * self.sup_parent_classes) for i in range(batch_size)], :, :, :]
        parent_map = F.interpolate(parent_map.reshape(batch_size, self.num_maps, 14, 14), size=(28, 28), mode='bilinear',
                                  align_corners=True)

        root_map = self.root_map.reshape(batch_size * self.sup_root_classes, self.num_maps, 28, 28).clone()
        root_map = root_map[[gt_root_label[i].long() + (i * self.sup_root_classes) for i in range(batch_size)], :, :, :]
        root_map = root_map.reshape(batch_size, self.num_maps, 28, 28)

        return torch.cat((child_map,parent_map, root_map),1)

    def get_gt_map_ce(self, gt_root_label, gt_parent_label, gt_child_label):
        batch_size = self.child_map_ce.size(0)
        child_map_ce = self.child_map_ce.reshape(batch_size*self.child_num_classes, self.num_maps, 14, 14).clone()
        child_map_ce = child_map_ce[[gt_child_label[i].long()+(i*self.child_num_classes) for i in range(batch_size)], :, :, :]
        child_map_ce = F.interpolate(child_map_ce.reshape(batch_size, self.num_maps, 14, 14),size=(28, 28), mode='bilinear', align_corners=True)

        if self.args.NoHDA:
            return child_map_ce

        parent_map_ce = self.parent_map_ce.reshape(batch_size * self.parent_num_classes, self.num_maps, 14, 14).clone()
        parent_map_ce = parent_map_ce[[gt_parent_label[i].long() + (i * self.parent_num_classes) for i in range(batch_size)], :, :, :]
        parent_map_ce = F.interpolate(parent_map_ce.reshape(batch_size, self.num_maps, 14, 14), size=(28, 28), mode='bilinear',
                                  align_corners=True)

        root_map_ce = self.root_map_ce.reshape(batch_size * self.root_num_classes, self.num_maps, 28, 28).clone()
        root_map_ce = root_map_ce[[gt_root_label[i].long() + (i * self.root_num_classes) for i in range(batch_size)], :, :, :]
        root_map_ce = root_map_ce.reshape(batch_size, self.num_maps, 28, 28)

        return torch.cat((child_map_ce,parent_map_ce, root_map_ce),1)


    def get_loss(self, logits, gt_root_label, gt_parent_label, gt_child_label,
                 child_h=None, child_v=None, parent_h=None, parent_v=None,
                 root_h=None, root_v=None, epoch=0, epoch_th=20):
        if self.args.lb:
            root_map, parent_map, child_map, root_map_ce, parent_map_ce, child_map_ce = logits
            root_map = torch.mean(root_map, dim=2)
            parent_map = torch.mean(parent_map, dim=2)
            child_map = torch.mean(child_map, dim=2)
            root_map_ce = torch.mean(root_map_ce, dim=2)
            parent_map_ce = torch.mean(parent_map_ce, dim=2)
            child_map_ce = torch.mean(child_map_ce, dim=2)
            if epoch > epoch_th:
                root_logits = self.get_logits(root_map, gt_root_label, self.args.cls_th)
                parent_logits = self.get_logits(parent_map, gt_parent_label, self.args.cls_th)
                child_logits = self.get_logits(child_map, gt_child_label, self.args.cls_th)
                root_logits_ce = self.get_logits(root_map_ce, gt_root_label, self.args.cls_th)
                parent_logits_ce = self.get_logits(parent_map_ce, gt_parent_label, self.args.cls_th)
                child_logits_ce = self.get_logits(child_map_ce, gt_child_label, self.args.cls_th)
            else:
                root_logits = torch.mean(torch.mean(root_map, dim=2), dim=2)
                parent_logits = torch.mean(torch.mean(parent_map, dim=2), dim=2)
                child_logits = torch.mean(torch.mean(child_map, dim=2), dim=2)
                root_logits_ce = torch.mean(torch.mean(root_map_ce, dim=2), dim=2)
                parent_logits_ce = torch.mean(torch.mean(parent_map_ce, dim=2), dim=2)
                child_logits_ce = torch.mean(torch.mean(child_map_ce, dim=2), dim=2)
        else:
            root_map, parent_map, child_map = logits
            root_map = torch.mean(root_map, dim=2)
            parent_map = torch.mean(parent_map, dim=2)
            child_map = torch.mean(child_map, dim=2)

            if epoch > epoch_th:
                root_logits = self.get_logits(root_map, gt_root_label, self.args.cls_th)
                parent_logits = self.get_logits(parent_map, gt_parent_label, self.args.cls_th)
                child_logits = self.get_logits(child_map, gt_child_label, self.args.cls_th)
            else:
                root_logits = torch.mean(torch.mean(root_map, dim=2), dim=2)
                parent_logits = torch.mean(torch.mean(parent_map, dim=2), dim=2)
                child_logits = torch.mean(torch.mean(child_map, dim=2), dim=2)



        # maps = torch.cat((
        #     F.interpolate((self.child_map.reshape(batch_size*self.child_num_classes, self.num_maps, 14, 14)[[gt_child_label[i].long()+(i*self.child_num_classes) for i in range(batch_size)], :, :, :]).reshape(batch_size, self.num_maps, 14, 14),size=(28, 28), mode='bilinear', align_corners=True),
        #     F.interpolate((self.parent_map.reshape(batch_size * self.parent_num_classes, self.num_maps, 14, 14)[[gt_parent_label[i].long() + (i * self.parent_num_classes) for i in range(batch_size)], :, :, :]).reshape(batch_size, self.num_maps, 14, 14), size=(28, 28),
        #                mode='bilinear', align_corners=True),
        #     (self.root_map.reshape(batch_size*self.root_num_classes, self.num_maps, 28, 28)[[gt_root_label[i].long() + (i * self.root_num_classes) for i in range(batch_size)], :, :, :]).reshape(batch_size, self.num_maps, 28, 28)), 1)
        maps = self.get_gt_map(gt_root_label, gt_parent_label, gt_child_label)
        if self.args.lb:
            maps_ce = self.get_gt_map_ce(gt_root_label, gt_parent_label, gt_child_label)
        loss_sim = None
        loss_cos, _ = self.calculate_cosineloss(maps)
        if self.args.lb:
            loss_cos_ce, _ = self.calculate_cosineloss(maps_ce)
        if self.args.mce:
            root_loss_cls = self.loss_cross_entropy(root_logits, gt_root_label.long())
            parent_loss_cls = self.loss_cross_entropy(parent_logits, gt_parent_label.long())
            child_loss_cls = self.loss_cross_entropy(child_logits, gt_child_label.long())
        elif self.args.bbce:
            if self.args.lb:
                root_loss_cls_ce = self.loss_cross_entropy(root_logits_ce, gt_root_label.long())
                parent_loss_cls_ce = self.loss_cross_entropy(parent_logits_ce, gt_parent_label.long())
                child_loss_cls_ce = self.loss_cross_entropy(child_logits_ce, gt_child_label.long())

            n = root_logits.size(0)
            gt_root = torch.zeros_like(root_logits)
            gt_root[:,-self.args.sup:] = 1
            gt_root[torch.arange(n),gt_root_label.long()] = 1

            gt_parent = torch.zeros_like(parent_logits)
            gt_parent[:, -self.args.sup:] =1
            gt_parent[torch.arange(n), gt_parent_label.long()] = 1

            gt_child = torch.zeros_like(child_logits)
            gt_child[:,-self.args.sup:] = 1
            gt_child[torch.arange(n), gt_child_label.long()] = 1
            if self.args.weight_bce:
                gt_root_w = torch.ones_like(root_logits)*(1-self.args.bce_pos_weight)/(root_logits.size(1)-1)
                gt_root_w[:,-self.args.sup:] = self.args.bce_pos_weight * (1-self.args.bbce_pos_weight)/self.args.sup
                gt_root_w[torch.arange(n), gt_root_label.long()] = self.args.bce_pos_weight *self.args.bbce_pos_weight

                gt_parent_w = torch.ones_like(parent_logits) * (1 - self.args.bce_pos_weight)/(parent_logits.size(1)-1)
                gt_parent_w[:,-self.args.sup:] = self.args.bce_pos_weight* (1-self.args.bbce_pos_weight)/self.args.sup
                gt_parent_w[torch.arange(n), gt_parent_label.long()] = self.args.bce_pos_weight *self.args.bbce_pos_weight

                gt_child_w = torch.ones_like(child_logits) * (1 - self.args.bce_pos_weight)/(child_logits.size(1)-1)
                gt_child_w[:,-self.args.sup:] = self.args.bce_pos_weight* (1-self.args.bbce_pos_weight)/self.args.sup
                gt_child_w[torch.arange(n), gt_child_label.long()] = self.args.bce_pos_weight*self.args.bbce_pos_weight

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
        elif self.args.bce:
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
            print('Wrong Loss function.')

        if self.args.lb:
            if self.args.NoHDA:
                loss_val = child_loss_cls * 0.5 + self.cos_alpha * loss_cos + \
                           self.cos_alpha * loss_cos_ce + child_loss_cls_ce * 0.5
            else:
                loss_bbce_val = root_loss_cls * 0.5 + parent_loss_cls * 0.5 + child_loss_cls * 0.5 + \
                                self.cos_alpha * loss_cos
                loss_ce_val = self.cos_alpha * loss_cos_ce  + root_loss_cls_ce * 0.5 + \
                              parent_loss_cls_ce * 0.5 + child_loss_cls_ce * 0.5
                loss_val = 2* loss_bbce_val * self.args.lb_bbce_weight + 2* loss_ce_val * (1.-self.args.lb_bbce_weight)
        else:
            if self.args.NoHDA:
                loss_val = child_loss_cls * 0.5 + self.cos_alpha * loss_cos
                if self.args.sc:
                    loss_sim = self.args.sc_alpha*self.get_cls_simliar_loss(gt_child_label.long(),*self.child_cls_fea, child_h.avg, child_v.avg)
                    loss_val += loss_sim
            else:
                loss_val = root_loss_cls*0.5 + parent_loss_cls*0.5 + child_loss_cls*0.5 + self.cos_alpha*loss_cos
                if self.args.sc:
                    child_loss_sim = self.args.sc_alpha * self.get_cls_simliar_loss(gt_child_label.long(),
                                                                              *self.child_cls_fea, child_h.avg,
                                                                              child_v.avg)
                    # parent_loss_sim = self.args.sc_alpha * self.get_cls_simliar_loss(gt_parent_label.long(),
                    #                                                                 *self.parent_cls_fea, parent_h.avg,
                    #                                                                 parent_v.avg)
                    # root_loss_sim = self.args.sc_alpha * self.get_cls_simliar_loss(gt_root_label.long(),
                    #                                                                 *self.root_cls_fea, root_h.avg,
                    #                                                                 root_v.avg)
                    # loss_sim = child_loss_sim + parent_loss_sim + root_loss_sim
                    loss_sim = child_loss_sim
                    loss_val += loss_sim

        return loss_val, root_loss_cls, parent_loss_cls, child_loss_cls, loss_cos, loss_sim,root_logits, parent_logits, child_logits


    def get_child_maps(self):
        return torch.mean(F.relu(self.child_map), dim=2)
        # return self.child_map

    def get_parent_maps(self):
        return torch.mean(F.relu(self.parent_map), dim=2)
        # return self.parent_map

    def get_root_maps(self):
        return torch.mean(F.relu(self.root_map), dim=2)
        # return self.root_map

    def get_child_maps_ce(self):
        return torch.mean(F.relu(self.child_map_ce), dim=2)
        # return self.child_map

    def get_parent_maps_ce(self):
        return torch.mean(F.relu(self.parent_map_ce), dim=2)
        # return self.parent_map

    def get_root_maps_ce(self):
        return torch.mean(F.relu(self.root_map_ce), dim=2)
        # return self.root_map
    def get_logits(self, cls_map, gt_label,cls_th):
        b, c, h, w = cls_map.size()
        cls_map = cls_map.contiguous().view(b,c,-1)
        cls_map_c = cls_map[torch.arange(b), gt_label.long()]
        min = torch.min(cls_map_c, dim=1, keepdim=True)[0]
        max = torch.max(cls_map_c, dim=1, keepdim=True)[0]
        norma_cls_map = (cls_map_c - min)/(max - min +1e-10)
        mask = norma_cls_map > cls_th
        logits = []
        for i_batch in range(b):
            logits.append(torch.mean(cls_map[i_batch, :,mask[i_batch]], dim=1).unsqueeze(0))
        return torch.cat(logits, dim=0)

class trunctable_cross_entropy(nn.Module):
    def __init__(self,threshold=0.6, softmax=True ):
        super(trunctable_cross_entropy, self).__init__()
        self.threshold = threshold
        self.softmax_flag = softmax

    def forward(self, logit, labels):
        n = logit.size(0)
        if self.softmax_flag:
            logit_ = F.softmax(logit,dim=1)
            pred_prop = logit_[torch.arange(n), labels.long()]
        else:
            logit_ = F.sigmoid(logit)
            pred_prop = logit_[torch.arange(n), torch.nonzero(labels==1)[:,1]]

        remain_sam = pred_prop <= self.threshold
        if torch.sum(remain_sam) == 0:
            return torch.Tensor([0.]).squeeze().cuda()
        if self.softmax_flag:
            loss = F.cross_entropy(logit[remain_sam,:],labels[remain_sam])
        else:
            loss = F.binary_cross_entropy(logit[remain_sam,:], labels[remain_sam,:])

        return loss

class residual_gap_block(nn.Module):
    def __init__(self,c_in, c_out, kernel, padding, dilation):
        super(residual_gap_block,self).__init__()
        # self.c_in = c_in
        # self.c_out = c_out
        # self.kernel = kernel
        # self.padding = padding
        # self.dilation = dilation
        self.conv = nn.Conv2d(c_in, c_out, kernel_size =kernel, padding=padding, dilation=dilation)
        self.residual = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        res = self.residual(x)
        x = self.relu(x + res)
        return x

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
    if kwargs['args'].IN:
        layers = make_layers(cfg['O'], dilation=dilation['D1'], instance_norm=True)
    elif kwargs['args'].INL:
        layers = make_layers(cfg['O'], dilation=dilation['D1'], instance_norm=True, inl=True)
    else:
        layers = make_layers(cfg['O'], dilation=dilation['D1'])
    cnv = np.cumsum(cnvs['OI']) if kwargs['args'].IN or kwargs['args'].INL else np.cumsum(cnvs['O'])
    model = VGG(layers, cnvs=cnv, **kwargs)
    if pretrained:
        if kwargs['args'].IN or kwargs['args'].INL:
            pred = [0,2,5,7]
            mod = [0,2,5,8]
            pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(j)) for i,j in zip(pred, mod)]
            pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(j)) for i,j in zip(pred,mod)]
        else:
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
