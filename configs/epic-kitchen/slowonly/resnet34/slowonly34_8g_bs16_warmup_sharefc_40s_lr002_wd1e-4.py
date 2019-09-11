# model settings

model = dict(
    type='TSN3D',
    backbone=dict(
        type='ResNet_SlowFast',
        pretrained='/home/yhxu/pretrain_model/resnet34_kinetics_backbone.pth',
        pretrained2d=False,
        depth=34,
        num_stages=4,
        out_indices=[3],
        frozen_stages=-1,
        inflate_freq=((0,0,0), (0,0,0,0), (1,1,1,1,1,1), (1,1,1)),
        inflate_style='3x1x1',
        conv1_kernel_t=1,
        conv1_stride_t=1,
        pool1_kernel_t=1,
        pool1_stride_t=1,
        bn_eval=False,
        partial_bn=True,
        style='pytorch'),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='avg',
        temporal_size=8,
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ShareClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.8,
        in_channels=512,
        hid_channels=512,
        noun_num_classes=352,
        verb_num_classes=125,
        init_std=0.001,
    )
)
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'KitchenDataset'
data_root = 'data/epic-kitchen/rawframes'
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.375, 57.375, 57.375], to_rgb=True)

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
    two_head=True,
    train=dict(
        type=dataset_type,
        ann_file='data/epic-kitchen/train.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=8,
        new_step=8,
        random_shift=True,
        modality='RGB',
        image_tmpl='frame_{:010d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=True,
        scales=[1, 0.875, 0.75, 0.66],
        max_distort=1,
        test_mode=False,
        ob_time=1.5,
        an_time=1),
    val=dict(
        type=dataset_type,
        ann_file='data/epic-kitchen/val.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=8,
        new_step=8,
        random_shift=False,
        modality='RGB',
        image_tmpl='frame_{:010d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=False,
        ob_time = 1.5,
        an_time = 1,
        ),
    test=dict(
        type=dataset_type,
        ann_file='data/epic-kitchen/val.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=3,
        new_length=8,
        new_step=8,
        random_shift=False,
        modality='RGB',
        image_tmpl='frame_{:010d}.jpg',
        img_scale=256,
        input_size=256,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='three_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True,
        ob_time=1.5,
        an_time=1))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1,
    step=[40, 60])
checkpoint_config = dict(interval=1)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tsn_kitchen_rgb_slowonly_8g_bs16_warmup_sharefc_ar_pretrained_no_frozen_40s'
load_from = None
resume_from = None



