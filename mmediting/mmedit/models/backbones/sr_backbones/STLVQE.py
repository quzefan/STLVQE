# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, ResidualBlocksWithInputConv2x2, SPyNet)
from mmedit.models.backbones.sr_backbones.deform_conv_v2 import DeformConv2d
from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

@BACKBONES.register_module()
class STLVQE(nn.Module):
    """BasicVSR++ network structure.

    Support either x4 upsampling or same size output.

    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_pretrained=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        # self.spynet = SPyNet(pretrained=spynet_pretrained)
        # self.spynet_distill = SPyNet(pretrained=spynet_pretrained)
        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(1, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                # nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                # nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['forward_1']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                3,
                mid_channels,
                2,
                padding=0,
                deform_groups=1,
                max_residue_magnitude=max_residue_magnitude)
            # self.backbone[module] = ResidualBlocksWithInputConv(
            #     (2 + i) * mid_channels, mid_channels, num_blocks)
            self.backbone[module] = ResidualBlocksWithInputConv(
                2 * mid_channels, mid_channels, num_blocks)

        self.conv_hr_first = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_last_first = nn.Conv2d(32, 1, 3, 1, 1)

        # upsampling module
        self.reconstruction1 = ResidualBlocksWithInputConv2x2(
            32, 32, 4)
        self.reconstruction2 = ResidualBlocksWithInputConv2x2(
            32, 32, 4)
        self.reconstruction3 = ResidualBlocksWithInputConv2x2(
            32, 32, 4)

        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        
        self.conv_last1 = nn.Conv2d(32, 1, 1, 1)
        self.conv_last2 = nn.Conv2d(32, 1, 1, 1)
        self.conv_last3 = nn.Conv2d(32, 1, 1, 1)

        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

    def propagate(self, feats, module_name, lqs):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """
        feats_rot = []
        weight = []
        t = len(feats['spatial'])
        frame_idx = range(0, t)
        # flow_idx = range(-1, t)     #错位
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        # feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            frame_current = lqs[:, i, :, :, :]
            # second-order deformable alignment
            if i > 0:
                frame_n1 = lqs[:, i - 1, :, :, :]
                frame_n2 = frame_n1
                feat_n2 = feat_prop

                if i > 1:  # second-order features
                    frame_n2 = lqs[:, i - 2, :, :, :]
                    feat_n2 = feats[module_name][-2]

                cond = torch.cat([feat_prop, feat_current, feat_n2], dim=1)
                frame_prop = torch.cat([frame_n1, frame_n2], dim=0)
                feat_prop, weights = self.deform_align[module_name](frame_prop, cond, frame_current, frame_is_1 = False)
                feats_rot.append(feat_prop)
                weight.append(weights)

            feat = feats['spatial'][mapping_idx[idx]]
            feat_prop = feat
            feats[module_name].append(feat)

        return feats_rot, weight
  
    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        # num_outputs = len(feats['spatial'])

        # mapping_idx = list(range(0, num_outputs))
        # mapping_idx += mapping_idx[::-1]
        frame_num = lqs.size(0)
        for i in range(0, lqs.size(1) - 1):
            hr_frame = feats[i]
            hr_all = 0
            hr_spatial = hr_frame[0]
            hr_temporal = hr_frame[1]
            hr_surround = hr_frame[2]
            hr_rot_curr_sum = 0
            hr_rot_temp_sum = 0
            for r in [0, 1, 2, 3]:
                hr_rot_curr = self.reconstruction1(hr_spatial[r])
                hr_rot_curr = self.conv_last1(hr_rot_curr)
                
                spatial_result = torch.rot90(hr_rot_curr, (4 - r) % 4, [2, 3])
                hr_rot_curr_sum = hr_rot_curr_sum + (spatial_result)
                
                hr_rot_temp = self.reconstruction2(hr_temporal[r])
                hr_rot_temp = self.conv_last2(hr_rot_temp)
                temp_result = hr_rot_temp
                hr_rot_temp_sum = hr_rot_temp_sum + (temp_result)
                
            hr_surround = self.reconstruction3(hr_surround)
            hr_surround = self.conv_last3(hr_surround)
            
            hr_surround_sum = (hr_surround[:frame_num] + hr_surround[frame_num:])

            hr_last = hr_rot_curr_sum / 4 + hr_rot_temp_sum / 4 + hr_surround_sum / 2
            
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i + 1, :, :, :])
            else:
                hr = hr_last
                # hr = hr.view(-1, 3, hr.size(2), hr.size(3))
                hr += lqs[:, i + 1, :, :, :]

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs, qp_value = None):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()

        feats = {}
        feats_ori = self.feat_extract(lqs.view(-1, c, h, w))
        h_down, w_down = feats_ori.shape[2:]
        feats_new = self.lrelu(self.upsample1(feats_ori))
        feats_new = self.lrelu(self.conv_hr_first(feats_new))
        residual_new = self.conv_last_first(feats_new).view(n, t, -1, h, w)
        lqs_new = residual_new + lqs
        feat = feats_ori.view(n, t, -1, h_down, w_down)
        feats['spatial'] = [feat[:, i, :, :, :] for i in range(0, t)]

        # feature propagation
        for iter_ in [1]:
            for direction in ['forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []
                feats, weight = self.propagate(feats, module, lqs_new)

        if self.training:
            return self.upsample(lqs_new, feats)
        else:
            return self.upsample(lqs_new, feats)
        
    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)
        
        self.deform_conv = DeformConv2d(1, 32, kernel_size=3, padding=1, stride=1, bias=None, modulation=False)
        self.quan_df = LqsQuan(bit = 128, s = 0.008)
        self.quan_temporal = LqsQuan(bit = 21, s = 0.04)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_feat = nn.Sequential(
            nn.Conv2d(3 * self.out_channels, 2 * self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * self.out_channels, 2 * self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * self.out_channels, 2 * self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            PixelShufflePack(2 * self.out_channels, 2 * self.out_channels, 2, upsample_kernel=3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            # nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * self.out_channels, 18 * self.deform_groups * 2 + 2, 3, 1, 1),
        )
        
        self.conv_current = nn.Conv2d(1, 32, 2, 1)
        self.conv_temporal = nn.Conv2d(3, 32, [2,1], stride = [2,1])
        self.conv_surround = nn.Conv2d(1, 32, 2, stride = 3, dilation = 2)
        
        self.softmax = nn.Softmax(dim = 1)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_feat[-1], val=0, bias=0)

    def forward(self, x, extra_feat = None, x_current = None, frame_is_1 = False):
        if frame_is_1 == False:
            feat = self.conv_feat(extra_feat)
            weights = self.softmax(feat[:, -2:])
            out = feat[:, :-2]
            o1, o2 = torch.chunk(out, 2, dim=1)
            offset = self.max_residue_magnitude * torch.tanh(
                torch.cat((o1, o2), dim=1))
            offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
            offset = torch.cat([offset_1, offset_2], dim=0)
            
            x_list = []
            x_offset = self.deform_conv(x, offset)

            nn_Unfold_current = nn.Unfold(kernel_size = (3,3), stride = 1, padding = 1)
            nn_Unfold = nn.Unfold(kernel_size = (3,3), stride = 3)
            x_current_deform = nn_Unfold_current(x_current).view(x_current.size(0), x_current.size(1), 3, 3, -1)

            x_offset_conv = nn_Unfold(x_offset).view(x.size(0), x.size(1), 3, 3, -1)
            b = x_offset_conv.size(0)
            x_offset_conv = torch.cat([x_offset_conv[:b//2], x_offset_conv[b//2:]], dim = 1)
            # print(x_offset_conv.shape)
            x_current_rot = []
            x_all = []
            # x_rot = torch.rot90(x_current, r, [2, 3]).cuda()
            for r in [0, 1, 2, 3]:
                x_rot = torch.rot90(x_current, r, [2, 3]).cuda()
                x_rot = F.pad(x_rot, (0, 1, 0, 1), mode='replicate')
                x_current_rot.append(x_rot.view(x_rot.size(0), 1, x_rot.size(2), x_rot.size(3)))

                nn_Fold_h = nn.Fold(output_size = (x_current.size(2) * 2, x_current.size(3)), kernel_size=(2, 1) ,stride = (2, 1))
                
                x_offset_conv_rot = torch.rot90(x_offset_conv, r, [2, 3])
                x_current_conv_rot = torch.rot90(x_current_deform, r, [2, 3])
                
                x_offset_conv_rot_down = x_offset_conv_rot[:,:,1:,1,:].reshape(x_offset_conv_rot.size(0), -1, x_offset_conv_rot.size()[-1])
                x_current_rot_down = x_current_conv_rot[:,:,1:,1,:].reshape(x_current_conv_rot.size(0), -1, x_current_conv_rot.size()[-1])
                
                x_offset_deconv_down = nn_Fold_h(x_offset_conv_rot_down)
                x_current_rot_down = nn_Fold_h(x_current_rot_down)
                
                x_offset_deconv_down = x_offset_deconv_down.view(x_offset_deconv_down.size(0), 2, x_offset_deconv_down.size(2), x_offset_deconv_down.size(3))
                x_current_rot_down = x_current_rot_down.view(x_current_rot_down.size(0), 1, x_current_rot_down.size(2), x_current_rot_down.size(3))
                
                x_offset_deconv_down = torch.cat([x_current_rot_down, x_offset_deconv_down], dim = 1)
                x_offset_deconv_down = self.quan_temporal(x_offset_deconv_down)
                x_offset_deconv_down = self.act(self.conv_temporal(x_offset_deconv_down))
                x_all.append(x_offset_deconv_down)
            
                x_current_rot[r] = self.quan_df(x_current_rot[r])
                x_current_rot[r] = self.act(self.conv_current(x_current_rot[r]))
            x_offset_surround = self.quan_df(x_offset)

            x_offset_deconv_surround = self.act(self.conv_surround(x_offset_surround))
            x_list.append(x_current_rot)
            x_list.append(x_all)
            x_list.append(x_offset_deconv_surround)

            return x_list, weights
        else:
            x_list = []
            x_current_rot = []
            # print(self.quan_df.s, self.quan_temporal.s)
            for r in [0, 1, 2, 3]:
                x_rot = torch.rot90(x, r, [2, 3]).cuda()
                x_rot = F.pad(x_rot, (0, 1, 0, 1), mode='replicate')
                x_current_rot.append(x_rot.view(x_rot.size(0), 1, x_rot.size(2), x_rot.size(3)))

            for r in [0, 1, 2, 3]:
                x_current_rot[r] = self.quan_df(x_current_rot[r])
                x_current_rot[r] = self.act(self.conv_current(x_current_rot[r]))
            x_all = x_current_rot
            x_list.append(x_all)
            return x_list

class LqsQuan(nn.Module):
    def __init__(self, bit, all_positive=True, symmetric=False, per_channel=True, s = 0.004):
        super().__init__()
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.s = nn.Parameter(torch.tensor(s)) 
    
    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad


    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad
    
    def forward(self, x):
        s_grad_scale = 1.0 / ((255 * x.numel()) ** 0.5)
        s_scale = self.grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)
        x = x * s_scale
        return x