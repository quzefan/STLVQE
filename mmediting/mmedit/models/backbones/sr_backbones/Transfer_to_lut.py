import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmedit.models.backbones.sr_backbones.basicvsr_pp import BasicVSRPlusPlus
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmedit.models import build_model
# from mmcv.parallel import MMDataParallel

MODEL_PATH = "./work_dirs/STLVQE/iter_100000.pth"    # Trained SR net params
SAMPLING_INTERVAL = 2        # N bit uniform sampling

model_set = dict(
    type='BasicVSR',
    generator=dict(
        type='STLVQE',
        mid_channels=32,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        cpu_cache_length=100),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    # ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
model = build_model(model_set, train_cfg=None, test_cfg=test_cfg)
device_id = torch.cuda.current_device()
load_checkpoint(model, "./work_dirs/STLVQE/temporal_quan_128_21/iter_100000.pth", map_location=lambda storage, loc: storage.cuda(device_id))
model_G = model.generator
model_G = model_G.cuda()

### Extract input-output pairs
with torch.no_grad():
    model_G.eval()
    act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    # 1D input
    # base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base = torch.arange(0, 128)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

    # 4D input
    fourth = base.unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfour = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfour.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2).float().cpu()
    input_surround = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], 3, 3, dtype = input_tensor.dtype)
    input_surround[:, :, 0, 0] = input_tensor[:, :, 0, 0]
    input_surround[:, :, 0, 2] = input_tensor[:, :, 0, 1]
    input_surround[:, :, 2, 0] = input_tensor[:, :, 1, 0]
    input_surround[:, :, 2, 2] = input_tensor[:, :, 1, 1]

    input_tensor_current = model_G.deform_align['forward_1'].quan_df.s.cpu() * input_tensor
    
    print(model_G.deform_align['forward_1'].quan_df.s, model_G.deform_align['forward_1'].quan_temporal.s)
    
    B = input_tensor.size(0) // 500
    outputs_current = []
    outputs_surround = []
    for b in range(500):
        if b == 499:
            batch_output_current = act(model_G.deform_align['forward_1'].conv_current(input_tensor_current[b*B:].cuda()))
            batch_output_current = model_G.reconstruction1(batch_output_current)
            # batch_output_current = model_G.lrelu(batch_output_current)
            batch_output_current = model_G.conv_last1(batch_output_current)
            
        else:
            batch_output_current = act(model_G.deform_align['forward_1'].conv_current(input_tensor_current[b*B:(b+1)*B].cuda()))
            batch_output_current = model_G.reconstruction1(batch_output_current)
            # batch_output_current = model_G.lrelu(batch_output_current)
            batch_output_current = model_G.conv_last1(batch_output_current)
            

        outputs_current += [ batch_output_current.cpu() ]
    
    results_current = np.concatenate(outputs_current, 0)
    print(np.mean(results_current))
    np.save("./work_dirs/STLVQE/temporal_quan_128_21/Model_Current", results_current)
    results_current = 0

    input_tensor_surround = model_G.deform_align['forward_1'].quan_df.s.cpu() * input_surround

    for b in range(500):
        if b == 499:    
            batch_output_surround = act(model_G.deform_align['forward_1'].conv_surround(input_tensor_surround[b*B:].cuda()))
            batch_output_surround = model_G.conv_last3(batch_output_surround)
        else:       
            batch_output_surround = act(model_G.deform_align['forward_1'].conv_surround(input_tensor_surround[b*B:(b+1)*B].cuda()))
            batch_output_surround = model_G.conv_last3(batch_output_surround)

        outputs_surround += [ batch_output_surround.cpu() ]

    results_surround = np.concatenate(outputs_surround, 0)
    print(np.mean(results_surround))
    print("Resulting LUT size: ", results_surround.shape)

    np.save("./work_dirs/STLVQE/temporal_quan_128_21/Model_Surround", results_surround)