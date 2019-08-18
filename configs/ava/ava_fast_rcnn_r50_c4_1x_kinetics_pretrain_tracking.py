# model settings
model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet_I3D',
        pretrained='open-mmlab://kin400/i3d_r50_f32s2_k400',
        pretrained2d=False,
        depth=50,
        num_stages=3,
        spatial_strides=(1, 2, 2),
        temporal_strides=(1, 1, 1),
        dilations=(1, 1, 1),
        out_indices=(2,),
        frozen_stages=-1,
        inflate_freq=((1,1,1), (1,0,1,0), (1,0,1,0,1,0)),
        inflate_style='3x1x1',
        conv1_kernel_t=5,
        conv1_stride_t=1,
        pool1_kernel_t=1,
        pool1_stride_t=1,
        bn_eval=False,
        partial_bn=False,
        bn_frozen=True,
        style='pytorch'),
    shared_head=dict(
        type='ResI3DLayer',
        pretrained='open-mmlab://kin400/i3d_r50_f32s2_k400',
        pretrained2d=False,
        depth=50,
        stage=3,
        spatial_stride=2,
        temporal_stride=1,
        dilation=1,
        style='pytorch',
        inflate_freq=(0, 1, 0),
        inflate_style='3x1x1',
        bn_eval=False,
        bn_frozen=True),
    bbox_roi_extractor=dict(
        type='SingleRoIStraight3DExtractorTracking',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=1024,
        featmap_strides=[16],
        with_temporal_pool=False),
    dropout_ratio=0.3,
    bbox_head=dict(
        type='BBoxHead',
        with_reg=False,
        with_temporal_pool=False,
        with_spatial_pool=True,
        spatial_pool_type='max',
        roi_feat_size=(1, 7, 7),
        in_channels=2048,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        multilabel_classification=True,
        reg_class_agnostic=True,
        nms_class_agnostic=True))
# model training and testing settings
train_cfg = dict(
    train_detector=False,
    person_det_score_thr=0.9,
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.9,
            neg_iou_thr=0.9,
            min_pos_iou=0.9,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=32, # 512,
            pos_fraction=1, #0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        cls_weight=1,
        debug=False))
test_cfg = dict(
    train_detector=False,
    person_det_score_thr=0.85,
    rcnn=dict(
        score_thr=-1, nms=dict(type='nms', iou_thr=1.0), max_per_img=100,
        action_thr=0.005))
# dataset settings
dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='data/ava/annotations/ava_train_v2.1.csv',
        exclude_file='data/ava/annotations/ava_train_excluded_timestamps_v2.1.csv',
        label_file='data/ava/annotations/ava_action_list_v2.1_for_activitynet_2018.pbtxt',
        video_stat_file='data/ava/ava_video_resolution_stats.csv',
        proposal_file='data/ava/tracking_ava_dense_proposals_train.FAIR.recall_93.9.pkl',
        # proposal_file = 'data/ava/train_tracking.pkl',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        new_length=32,
        new_step=2,
        random_shift=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=[(625, 200), ],
        input_size=None, # do no crop
        div_255=False,
        size_divisor=32,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        test_mode=False,
        with_label=True,
        with_tracking=True),
    val=dict(
        type=dataset_type,
        ann_file='data/ava/annotations/ava_val_v2.1.csv',
        exclude_file='data/ava/annotations/ava_val_excluded_timestamps_v2.1.csv',
        # exclude_file='data/ava/annotations/ava_train_excluded_timestamps_v2.1.csv',
        label_file='data/ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt',
        video_stat_file='data/ava/ava_video_resolution_stats.csv',
        # proposal_file='data/dense_proposals_val.recall_92.pkl',
        proposal_file='data/ava/tracking_ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        new_length=32,
        new_step=2,
        random_shift=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=[(625, 200), ],
        input_size=None,
        div_255=False,
        size_divisor=32,
        flip_ratio=0,
        resize_keep_ratio=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file='data/ava/annotations/ava_val_v2.1.csv',
        exclude_file='data/ava/annotations/ava_val_excluded_timestamps_v2.1.csv',
        label_file='data/ava/annotations/ava_action_list_v2.1_for_activitynet_2018.pbtxt',
        video_stat_file='data/ava/ava_video_resolution_stats.csv',
        proposal_file='data/ava/tracking_ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        img_prefix=data_root,
        img_norm_cfg=img_norm_cfg,
        input_format='NCTHW',
        new_length=32,
        new_step=2,
        random_shift=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=[(625, 200), ],
        input_size=None,
        div_255=False,
        size_divisor=32,
        flip_ratio=0,
        resize_keep_ratio=True,
        with_label=False,
        test_mode=True,
        with_tracking=True))
# optimizer
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=1e-6)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 4,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ava_fast_rcnn_r50_c4_1x_kinetics_pretrain_tracking'
load_from = None
resume_from = 'work_dirs/ava_fast_rcnn_r50_c4_1x_kinetics_pretrain_tracking/epoch_10.pth'
workflow = [('train', 1)]
