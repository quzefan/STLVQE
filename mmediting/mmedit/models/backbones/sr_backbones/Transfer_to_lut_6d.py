import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from mmedit.models.backbones.sr_backbones.basicvsr_pp import STLVQE
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmedit.models import build_model
# from mmcv.parallel import MMDataParallel

MODEL_PATH = "./work_dirs/STLVQE/temporal_quan_21/iter_15000.pth"    # Trained SR net params
SAMPLING_INTERVAL = 6        # N bit uniform sampling

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
load_checkpoint(model, "/mnt/Data/qzf/mmediting/work_dirs/LUT/temporal_quan_128_21/iter_200000.pth", map_location=lambda storage, loc: storage.cuda(device_id))
model_G = model.generator
model_G = model_G.cuda()

### Extract input-output pairs
with torch.no_grad():
    model_G.eval()
    act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    # 1D input
    # base = torch.arange(0, 257, 2**SAMPLING_INTERVAL)   # 0-256
    base = torch.arange(0, 21)   # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)                             # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L*L).reshape(-1) # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)    # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfour = torch.cat([fourth.unsqueeze(1), onebythree], 1)    # [256*256*256*256, 4]
    
    # 5D input
    fifth = base.cuda().unsqueeze(1).repeat(1, L*L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebyfour = onebyfour.repeat(L, 1)
    onebyfive = torch.cat([fifth.unsqueeze(1), onebyfour], 1)    # [256*256*256*256, 4]
    
    # 6D input
    sixth = base.cuda().unsqueeze(1).repeat(1, L*L*L*L*L).reshape(-1) # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebyfive = onebyfive.repeat(L, 1)
    onebysix = torch.cat([sixth.unsqueeze(1), onebyfive], 1)    # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_h = onebysix.unsqueeze(1).unsqueeze(1).reshape(-1,3,2,1).float()
    input_tensor_down = model_G.deform_align['forward_1'].quan_temporal.s * input_h
    # Split input to not over GPU memory
    B = input_h.size(0) // 100
    outputs_current = []
    outputs_top = []
    outputs_down = []
    outputs_left = []
    outputs_right = []
    for b in range(100):
        if b == 99:
            batch_output_down = act(model_G.deform_align['forward_1'].conv_temporal(input_tensor_down[b*B:]))
            batch_output_down = model_G.reconstruction2(batch_output_down)
            batch_output_down = model_G.conv_last2(batch_output_down)
        else:
            batch_output_down = act(model_G.deform_align['forward_1'].conv_temporal(input_tensor_down[b*B:(b+1)*B]))
            batch_output_down = model_G.reconstruction2(batch_output_down)
            batch_output_down = model_G.conv_last2(batch_output_down)
            
        outputs_down += [ batch_output_down.cpu() ]
    
    results_down = np.concatenate(outputs_down, 0)
    print(results_down[17*21*21*21*21*21 + 17*21*21*21*21 + 17*21*21*21 + 17*21*21 + 17*21 + 1], results_down[18*21*21*21*21*21 + 18*21*21*21*21 + 18*21*21*21 + 18*21*21 + 18*21 + 1])
    print("Resulting LUT size: ", results_down.shape)

    np.save("./work_dirs/STLVQE/temporal_quan_128_21/Model_Down_21upper", results_down)
