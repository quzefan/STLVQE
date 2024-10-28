exp_name = 'STLVQE'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='STLVQE_LUT',
        mid_channels=32,
        num_blocks=7,
        is_low_res_input=False,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        # spynet_pretrained='/mnt/Data/qzf/FastFlowNet/checkpoints/fastflownet_ft_mix.pth',
        cpu_cache_length=100),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    # ensemble=dict(type='SpatialTemporalEnsemble', is_temporal_ensemble=False),
)
# model training and testing settings
train_cfg = dict(fix_iter=0)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)
# dataset settings
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1], start_idx=1,
        filename_tmpl='{:04d}.png'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        convert_to='y',),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        convert_to='y',),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=180),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
# dataset settings
test_dataset_type = 'SRFolderMultipleGTDataset'
test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        start_idx=1,
        filename_tmpl='{:03d}.png'
        ),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        convert_to='y',),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        convert_to='y',),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu = 6,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True, workers_per_gpu=4),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1,workers_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='your/train_data/path',
            gt_folder='your/train_data_GT/path',
            num_input_frames=24,
            pipeline=train_pipeline,
            scale=1,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='your/valid_data/path',
        gt_folder='your/valid_data_GT/path',
        num_input_frames=24,
        pipeline=test_pipeline,
        scale=1,
        test_mode=True),

    test=dict(
        type=test_dataset_type,
        lq_folder='your/test_data/path',
        gt_folder='your/test_data_GT/path',
        pipeline=test_pipeline,
        scale=1,
        test_mode=True),
)
# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99)))

# learning policy
total_iters = 200000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[200000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=1500, save_image=False)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/STLVQE/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
