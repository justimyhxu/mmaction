# model settings
model = dict(
    type='TSN2D_ATTEN',
    backbone=dict(
        type='BNInception',
        pretrained='open-mmlab://bninception_caffe',
        bn_eval=False,
        partial_bn=True),
    spatial_temporal_module=dict(
        type='SimpleSpatialModule',
        spatial_type='avg',
        spatial_size=7),
    verb_segmental_consensus=dict(
        type='normal_Atten',
        time_scales=[1, 3, 5, 7, 15],
        in_channels=512,
        time_dim=16,
        num_classes=125),
    noun_segmental_consensus=dict(
        type='normal_Atten',
        time_scales=[1, 3, 5, 7, 15],
        in_channels=512,
        time_dim=16,
        num_classes=352),
    cls_head=dict(
        type='ShareClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.8,
        in_channels=1024,
        hid_channels=1024,
        noun_num_classes=352,
        verb_num_classes=125,
        init_std=0.001,
        ),
    verb_code=1,
    noun_code=1,
        )
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'KitchenDataset'
data_root = 'data/epic-kitchen/rawframes'
img_norm_cfg = dict(
   mean=[104, 117, 128], std=[1, 1, 1], to_rgb=False)

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='data/epic-kitchen/train.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=8,
        new_length=1,
        new_step=1,
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
        ob_time=1,
        an_time=1,
        with_pos_encoding=True),
    val=dict(
        type=dataset_type,
        ann_file='data/epic-kitchen/val.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=3,
        new_length=1,
        new_step=1,
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
        ob_time = 1,
        an_time = 1,
        ),
    test=dict(
        type=dataset_type,
        ann_file='data/epic-kitchen/val.txt',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        num_segments=8,
        new_length=1,
        new_step=1,
        random_shift=False,
        modality='RGB',
        image_tmpl='frame_{:010d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='ten_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True,
        ob_time=1,
        an_time=1,
        with_pos_encoding=True))
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.1,
    step=[30, 60])
checkpoint_config = dict(interval=1)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tsn_kitchen_rgb_bninception_8g_bs32_warmup_sharefc_seg8_atten'
load_from = None
resume_from = None



