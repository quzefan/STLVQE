# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import constant_init
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.runner import load_checkpoint
import quadrilinear4d
import quadrilinear6d
import time

from mmedit.models.backbones.sr_backbones.basicvsr_net import (
    ResidualBlocksWithInputConv, ResidualBlocksWithInputConv2x2, SPyNet)
from mmedit.models.backbones.sr_backbones.deform_conv_v2 import DeformConv2d
from mmedit.models.common import PixelShufflePack, flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.backbones.sr_backbones.modulated_deform import modulated_deform_conv2d
import torch.multiprocessing as mp
# USER PARAMS
SAMPLING_INTERVAL_4D = 2        # N bit uniform sampling
SAMPLING_INTERVAL_6D = 4        # N bit uniform sampling
# q = 2**SAMPLING_INTERVAL
Current_LUT_PATH = "./work_dirs/STLVQE/temporal_quan_128_21/Model_Current.npy"    # Trained SR net params
Down_LUT_PATH = "./work_dirs/STLVQE/temporal_quan_128_21/Model_Down_21upper.npy"    # Trained SR net params
Surround_LUT_PATH = "./work_dirs/STLVQE/temporal_quan_128_21/Model_Surround.npy"    # Trained SR net params
Tri_index_PATH = "./work_dirs/STLVQE/triangular_index.npy"
LUT_Current = torch.from_numpy(np.load(Current_LUT_PATH).astype(np.float32)).cuda()
LUT_Down = torch.from_numpy(np.load(Down_LUT_PATH).astype(np.float32)).cuda()
LUT_Surround = torch.from_numpy(np.load(Surround_LUT_PATH).astype(np.float32)).cuda()
Tri_index = torch.from_numpy(np.load(Tri_index_PATH).astype(np.float32)).cuda().flatten()

# MODEL_PATH = "./work_dirs/STLVQE/temporal_quan_128_21/iter_200000.pth"    # Trained SR net params
@BACKBONES.register_module()
class STLVQE_LUT(nn.Module):
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
        # self.backbone = nn.ModuleDict()
        modules = ['forward_1']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                3,
                mid_channels,
                2,
                padding=0,
                deform_groups=1,
                max_residue_magnitude=max_residue_magnitude)

        self.conv_hr_first = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_last_first = nn.Conv2d(32, 1, 3, 1, 1)
        
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_last1 = nn.Conv2d(32, 1, 1, 1)
        self.conv_last2 = nn.Conv2d(48, 1, 1, 1)
        self.conv_last3 = nn.Conv2d(32, 1, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

    def propagate(self, feats, feats_pre_1, feats_pre_2, lqs, lqs_pre_1, lqs_pre_2, frame_num):
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
        module_name = 'forward_1'
        if frame_num == 0:
            feat_prop = self.deform_align[module_name](lqs, frame_is_1 = True)
            return feat_prop
        cond_n1 = feats_pre_1
        
        if frame_num == 1:
            lqs_pre_2 = lqs_pre_1        
            cond_n2 = cond_n1
        
        else:
            cond_n2 = feats_pre_2
        
        cond = torch.cat([cond_n1, feats, cond_n2], dim=1)
        frame_prop = torch.cat([lqs_pre_1, lqs_pre_2], dim=0)
        feat_prop = self.deform_align[module_name](frame_prop, cond, lqs, frame_is_1 = False)
        
        
        return feat_prop
        
        

    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad


    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad
    
    def quan(self, x):
        s_grad_scale = 1.0 / ((255 * x.numel()) ** 0.5)
        s_scale = self.grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)
        x = x * s_scale
        return x

    
    def upsample(self, lqs, feats, frame_num):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        hr_frame = feats
        batch_size = lqs.size(0)
        if frame_num == 0:
            hr_rot_curr_sum = 0
            for r in [0, 1, 2, 3]:
                hr_rot_current = hr_frame[0][r].float()

                batch = hr_rot_current.size(0)
                h = hr_rot_current.size(2) - 1
                w = hr_rot_current.size(3) - 1
                x1 = hr_rot_current[:, :, 0:0+h, 0:0+w].flatten()
                x2 = hr_rot_current[:, :, 0:0+h, 1:1+w].flatten()
                x3 = hr_rot_current[:, :, 1:1+h, 0:0+w].flatten()
                x4 = hr_rot_current[:, :, 1:1+h, 1:1+w].flatten()
                hr_rotation_1 = torch.zeros_like(x1).cuda()
                weight = torch.ones([1]).cuda()

                assert 1 == quadrilinear4d.forward(LUT_Current, Tri_index, weight, x1, x2, x3, x4, hr_rotation_1, 1, 1, 1, 128, w, h, batch)
                
                spatial_result = torch.rot90(hr_rotation_1.view(-1, 1, h, w), (4 - r) % 4, [2, 3])
                hr_rot_curr_sum = hr_rot_curr_sum + spatial_result
            
            hr_last = (hr_rot_curr_sum / 4).view(-1, 1, hr_rot_curr_sum.size(2), hr_rot_curr_sum.size(3))

        else:
            hr_rot_curr_sum = 0
            hr_rot_temp_sum = 0
            # time_start = time.time() #开始计时
            hr_spatial = hr_frame[0]
            hr_temporal = hr_frame[1]
            hr_surround = hr_frame[2].float()
            weight = torch.ones([1]).cuda()
            hr_rot_current = [hr_spatial[r].float().cuda() for r in [0, 1, 2, 3]]
            hr_rot_down = [hr_temporal[r].float().cuda() for r in [0, 1, 2, 3]]

            for r in [0, 1, 2, 3]:
                batch = hr_rot_current[r].size(0)
                h = hr_rot_current[r].size(2) - 1
                w = hr_rot_current[r].size(3) - 1
                x1 = hr_rot_current[r][:, :, 0:0+h, 0:0+w].flatten()
                x2 = hr_rot_current[r][:, :, 0:0+h, 1:1+w].flatten()
                x3 = hr_rot_current[r][:, :, 1:1+h, 0:0+w].flatten()
                x4 = hr_rot_current[r][:, :, 1:1+h, 1:1+w].flatten()
                hr_rotation_1 = x1.cuda()
                assert 1 == quadrilinear4d.forward(LUT_Current, Tri_index, weight, x1, x2, x3, x4, hr_rotation_1, 1, 1, 1, 128, w, h, batch)
                
                spatial_result = torch.rot90(hr_rotation_1.view(-1, 1, h, w), (4 - r) % 4, [2, 3])
                hr_rot_curr_sum = hr_rot_curr_sum + spatial_result
                
                h0 = hr_rot_curr_sum.size(2)
                w0 = hr_rot_curr_sum.size(3)

                h = hr_rot_down[r][:batch_size].size(2)
                x1_down1 = hr_rot_down[r][:batch_size, 0, 0:0+h:2, :].flatten()
                x2_down1 = hr_rot_down[r][:batch_size, 0, 1:1+h:2, :].flatten()
                x3_down1 = hr_rot_down[r][:batch_size, 1, 0:0+h:2, :].flatten()
                x4_down1 = hr_rot_down[r][:batch_size, 1, 1:1+h:2, :].flatten()
                x5_down1 = hr_rot_down[r][:batch_size, 2, 0:0+h:2, :].flatten()
                x6_down1 = hr_rot_down[r][:batch_size, 2, 1:1+h:2, :].flatten()
                hr_rotation_4 = x1_down1.cuda()
                assert 1 == quadrilinear6d.forward(LUT_Down, Tri_index, weight, x1_down1, x2_down1, x3_down1, x4_down1, x5_down1, x6_down1, hr_rotation_4, 1, 1, 1, 21, w0, h0, batch)
                
                temporal_result = hr_rotation_4
                temporal_result = temporal_result.view(-1, 1, h0, w0)
                
                hr_rot_temp_sum = hr_rot_temp_sum + temporal_result

            hr_surround_1 = hr_surround[0:batch_size]
            hr_surround_2 = hr_surround[batch_size:]
            h = hr_surround_1.size(2)
            w = hr_surround_1.size(3)
            x1_surround1 = hr_surround_1[:, :, 0:0+h:3, 0:0+w:3].flatten()
            x2_surround1 = hr_surround_1[:, :, 0:0+h:3, 2:2+w:3].flatten()
            x3_surround1 = hr_surround_1[:, :, 2:2+h:3, 0:0+w:3].flatten()
            x4_surround1 = hr_surround_1[:, :, 2:2+h:3, 2:2+w:3].flatten()
            hr_rotation_6 = x1_surround1.cuda()
            assert 1 == quadrilinear4d.forward(LUT_Surround, Tri_index, weight, x1_surround1, x2_surround1, x3_surround1, x4_surround1, hr_rotation_6, 1, 1, 1, 128, w0, h0, batch)
            
            x1_surround2 = hr_surround_2[:, :, 0:0+h:3, 0:0+w:3].flatten()
            x2_surround2 = hr_surround_2[:, :, 0:0+h:3, 2:2+w:3].flatten()
            x3_surround2 = hr_surround_2[:, :, 2:2+h:3, 0:0+w:3].flatten()
            x4_surround2 = hr_surround_2[:, :, 2:2+h:3, 2:2+w:3].flatten()
            hr_rotation_7 = x1_surround1
            assert 1 == quadrilinear4d.forward(LUT_Surround, Tri_index, weight, x1_surround2, x2_surround2, x3_surround2, x4_surround2, hr_rotation_7, 1, 1, 1, 128, w0, h0, batch)
            
            hr_last = hr_rot_curr_sum / 4 + hr_rot_temp_sum / 4 + ((hr_rotation_6 + hr_rotation_7) / 2).view(hr_rot_temp_sum.size(0), hr_rot_temp_sum.size(1), hr_rot_temp_sum.size(2), hr_rot_temp_sum.size(3))

        # print(hr_rot_curr_sum[0, 0, 200:210, 200:210])
        # print(hr_rot_temp_sum[0, 0, 300:310, 200:210])
        # print((hr_rotation_6 + hr_rotation_7) / 2)
        # print("...")
        hr = lqs + hr_last


        return hr

    def forward(self, lqs, qp_value = None): 
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()
        feat_previous_1 = 0
        feat_previous_2 = 0
        lqs_new_previous_1 = 0
        lqs_new_previous_2 = 0
        lqs = lqs.half()
        feats = {}
        result_per_frame = []
        n, t, c, h, w = lqs.size()
        # compute spatial features
        for frame in range(0,t):
            lqs_current = lqs[:, frame].view(-1, 1, h, w).cuda().half()
            feats_ = self.feat_extract(lqs_current)
            
            if frame == 0:
                feats_new = self.lrelu(self.upsample1(feats_))
                feats_new = self.lrelu(self.conv_hr_first(feats_new))
                residual_new = self.conv_last_first(feats_new).view(n, 1, h, w)
                # residual_new = self.spatial_complement(feats_).view(n, -1, h, w)
                lqs_new = residual_new + lqs_current
                feats = self.deform_align['forward_1'](lqs_new, extra_feat=None, frame_is_1 = True)
                feat_previous_1 = feats_
                lqs_new_previous_1 = lqs_new
                continue

            # cond_n1 = feat_previous_1
            
            # if frame == 1:
            #     lqs_new_previous_2 = lqs_new_previous_1       
            #     cond_n2 = cond_n1
            
            # else:
            #     cond_n2 = feat_previous_2
            
            # cond = torch.cat([cond_n1, feats_, cond_n2], dim=1)
            # frame_prop = torch.cat([lqs_new_previous_1, lqs_new_previous_2], dim=0)
            
            feats_new = self.lrelu(self.upsample1(feats_))
            feats_new = self.lrelu(self.conv_hr_first(feats_new))
            residual_new = self.conv_last_first(feats_new).view(n, 1, h, w)
            lqs_new = lqs_current + residual_new
            
            for iter_ in [1]:
                for direction in ['forward']:
                    module = f'{direction}_{iter_}'
                    feats = self.propagate(feats_, feat_previous_1, feat_previous_2, lqs_new, lqs_new_previous_1, lqs_new_previous_2, frame)
            result_per_frame.append(self.upsample(lqs_new, feats, frame).unsqueeze(1))
            feat_previous_2 = feat_previous_1
            feat_previous_1 = feats_
            lqs_new_previous_2 = lqs_new_previous_1
            lqs_new_previous_1 = lqs_new

        result = torch.cat(result_per_frame, dim = 1)
        return result

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
        
        self.quan_df = LqsQuan(bit = 128, s = 0.01)
        self.quan_temporal = LqsQuan(bit = 21, s = 0.04)
        self.weights = torch.ones(1,1,3,3,device='cuda',requires_grad=False)
        self.biass = torch.zeros(1,device='cuda',requires_grad=False)
        self.zero_padding = nn.ZeroPad2d(1)

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
        self.nn_Unfold_current = nn.Unfold(kernel_size = (3,3), stride = 1, padding = 1)
        self.nn_Unfold = nn.Unfold(kernel_size = (3,3), stride = 3)
        self.p0 = 0
        self.pn = 0
        self.init_offset()

    def init_offset(self):
        return
        # constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, x_current = None, frame_is_1 = False):
        if frame_is_1 == True:
            x_list = []
            
            h, w = x.size(2), x.size(3)
            x_current_rot = []
            for r in [0, 1, 2, 3]:
                x_rot = torch.rot90(x, r, [2, 3])
                x_rot = F.pad(x_rot, (0, 1, 0, 1), mode='replicate')
                x_current_rot.append(self.quan_df(x_rot))
                
            x_list.append(x_current_rot)
            
            return x_list

        feat = self.conv_feat(extra_feat)
        out = feat[:, :-2]
        o1, o2 = torch.chunk(out, 2, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        x_list = []
        x_offset = modulated_deform_conv2d(x.repeat(1, 3, 1, 1), offset, self.weights, self.biass, 1, 1, 1, 1, 1, 64)[:, 0].unsqueeze(1)
        b = x_offset.size(0)
        x_offset_conv = torch.cat([x_offset[:b//2], x_offset[b//2:]], dim = 1)

        x_current_deform = self.nn_Unfold_current(x_current).view(x_current.size(0), x_current.size(1), 3, 3, -1)

        x_offset_conv = self.nn_Unfold(x_offset_conv).view(x_offset_conv.size(0), x_offset_conv.size(1), 3, 3, -1)
        
        x_rot = [self.quan_df(F.pad(torch.rot90(x_current, r, [2, 3]), (0, 1, 0, 1))) for r in range(4)]  

        x_offset_conv_rot = [torch.rot90(x_offset_conv, r, [2, 3]) for r in [0, 1, 2, 3]]
        x_current_conv_rot = [torch.rot90(x_current_deform, r, [2, 3]) for r in [0, 1, 2, 3]]
        
        x_offset_conv_rot_down = [(x_offset_conv_rot[r][:,:,1:,1,:].reshape(x_offset_conv_rot[r].size(0), -1, x_offset_conv_rot[r].size()[-1])) for r in range(4)]
        x_current_rot_down = [(x_current_conv_rot[r][:,:,1:,1,:].reshape(x_current_conv_rot[r].size(0), -1, x_current_conv_rot[r].size()[-1])) for r in range(4)]
        
        # nn_Fold_h = [nn.Fold(output_size = ((x_rot[r].size(2) - 1) * 2, x_rot[r].size(3) - 1), kernel_size=(2, 1) ,stride = (2, 1)) for r in [0, 1, 2, 3]]
        nn_Fold_h = nn.Fold(output_size = (x_current.size(2) * 2, x_current.size(3)), kernel_size=(2, 1) ,stride = (2, 1))
                
        x_offset_deconv_down = [self.quan_temporal(torch.cat([nn_Fold_h(x_current_rot_down[r]), nn_Fold_h(x_offset_conv_rot_down[r])], dim = 1)) for r in range(4)]

        x_list.append(x_rot)
        x_list.append(x_offset_deconv_down)
        x_list.append(self.quan_df(x_offset))
        
        return x_list
        

class LqsQuan(nn.Module):
    def __init__(self, bit, all_positive=True, symmetric=False, per_channel=True, s = 0.004):
        super().__init__()
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.thd_neg = 0
            self.thd_pos = bit - 1
        else:
            if symmetric:
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
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
        # x = x * s_scale
        return x