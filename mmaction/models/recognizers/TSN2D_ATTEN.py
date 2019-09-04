import torch.nn as nn
from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS
from .LearnWeights import LearnWeights
from .EncodePosition import EncodePosition
import torch.nn.functional as F

@RECOGNIZERS.register_module
class TSN2D_ATTEN(BaseRecognizer):

    def __init__(self,
                 backbone,
                 modality='RGB',
                 in_channels=3,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 verb_encode=None,
                 noun_encode=None,
                 verb_segmental_consensus=None,
                 noun_segmental_consensus=None,
                 verb_cls_head=None,
                 noun_cls_head=None,
                 with_LRW=False):

        super(TSN2D_ATTEN, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.modality = modality
        self.in_channels = in_channels
        self.with_LRW = with_LRW
        if self.with_LRW:
            self.LRW = LearnWeights()

        if verb_encode is not None:
            self.verb_encode = EncodePosition(in_dim = 60, h_dim=512, num_classes=125)
        if noun_encode is not None:
            self.noun_encode = EncodePosition(in_dim = 60, h_dim=512, num_classes=352)

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(
                spatial_temporal_module)
        else:
            pass

        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(
                segmental_consensus)
        else:
            pass
        if verb_segmental_consensus is not None:
            self.verb_segmental_consensus = builder.build_segmental_consensus(verb_segmental_consensus)

        if noun_segmental_consensus is not None:
            self.noun_segmental_consensus = builder.build_segmental_consensus(noun_segmental_consensus)

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        if verb_cls_head is not None:
            self.verb_cls_head = builder.build_head(verb_cls_head)
        if noun_cls_head is not None:
            self.noun_cls_head = builder.build_head(noun_cls_head)
        else:
            pass

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.init_weights()

        if modality == 'Flow' or modality == 'RGBDiff':
            self._construct_2d_backbone_conv1(in_channels)



    @property
    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    @property
    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None
    @property
    def with_noun_segmental_consensus(self):
        return hasattr(self, 'noun_segmental_consensus') and self.noun_segmental_consensus is not None
    @property
    def with_verb_segmental_consensus(self):
        return hasattr(self, 'verb_segmental_consensus') and self.verb_segmental_consensus is not None
    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @property
    def with_verb_head(self):
        return hasattr(self, 'verb_cls_head') and self.verb_cls_head is not None
    @property
    def with_noun_head(self):
        return hasattr(self, 'noun_cls_head') and self.noun_cls_head is not None
    @property
    def with_encode_verb(self):
        return hasattr(self, 'verb_encode') and self.verb_encode is not None
    @property
    def with_encode_noun(self):
        return hasattr(self, 'noun_encode') and self.noun_encode is not None


    def _construct_2d_backbone_conv1(self, in_channels):
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(
            modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (in_channels, ) + kernel_size[2:]
        new_kernel_data = params[0].data.mean(dim=1, keepdim=True).expand(
            new_kernel_size).contiguous()  # make contiguous!

        new_conv_layer = nn.Conv2d(in_channels, conv_layer.out_channels,
                                   conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                                   bias=True if len(params) == 2 else False)
        new_conv_layer.weight.data = new_kernel_data
        if len(params) == 2:
            new_conv_layer.bias.data = params[1].data
        # remove ".weight" suffix to get the layer layer_name
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv_layer)

    def init_weights(self):
        super(TSN2D_ATTEN, self).init_weights()
        self.backbone.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_noun_segmental_consensus:
            self.noun_segmental_consensus.init_weights()

        if self.with_verb_segmental_consensus:
            self.verb_segmental_consensus.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label,
                      **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        pos_encode = kwargs['pos_encode']

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        # x = x.reshape((-1, num_seg) + x.shape[1:])
        # if self.with_segmental_consensus:
        #     x = self.segmental_consensus(x)
        #     x = x.squeeze(1)
        losses = dict()
        if self.with_cls_head:
            verb_cls_score, noun_cls_score = self.cls_head(x)
            noun_cls_score_logit = noun_cls_score.reshape((-1, num_seg) + noun_cls_score.shape[1:])
            verb_cls_score_logit = verb_cls_score.reshape((-1, num_seg)+verb_cls_score.shape[1:])

            noun_init_logit, noun_final_logit = self.noun_segmental_consensus(pos_encode, noun_cls_score_logit)
            verb_init_logit, verb_final_logit = self.verb_segmental_consensus(pos_encode, verb_cls_score_logit)

            noun_gt_label = gt_label['noun'].squeeze()
            verb_gt_label = gt_label['verb'].squeeze()
            loss_score_noun = F.cross_entropy(noun_init_logit, noun_gt_label)
            loss_score_verb = F.cross_entropy(verb_init_logit, verb_gt_label)
            # print('ccccc')
        if self.with_noun_head:
            noun_cls_score = self.noun_cls_head(x)
            noun_cls_score_logit = noun_cls_score.reshape((-1, num_seg)+noun_cls_score.shape[1:])
            # noun_cls_score = noun_cls_score_logit.mean(dim=1)
            noun_init_logit, noun_final_logit = self.noun_segmental_consensus(pos_encode, noun_cls_score_logit)
            noun_gt_label = gt_label['noun'].squeeze()
            loss_score_noun = self.noun_cls_head.loss(noun_init_logit, noun_gt_label, name='noun_cls_loss')
            # noun_logit = loss_score_noun.reshape((-1, num_seg)+loss_score_noun.shpae[1:])
            # losses.update(loss_score_noun)


        if self.with_verb_head:
            verb_cls_score = self.verb_cls_head(x)
            verb_cls_score_logit = verb_cls_score.reshape((-1, num_seg)+verb_cls_score.shape[1:])
            # verb_cls_score = verb_cls_score_logit.mean(dim=1)
            verb_init_logit, verb_final_logit = self.verb_segmental_consensus(pos_encode, verb_cls_score_logit)
            verb_gt_label = gt_label['verb'].squeeze()
            loss_score_verb = self.verb_cls_head.loss(verb_init_logit, verb_gt_label, name='verb_cls_loss')
            # verb_logit = loss_score_verb.reshape((-1, num_seg)+loss_score_verb.shape[1:])
            # losses.update(loss_score_verb)


        if self.with_LRW:
            losses.update(dict(noun_cls_socre=loss_score_noun))
            losses.update(dict(verb_cls_score=loss_score_verb))
            loss = self.LRW(loss_score_noun,loss_score_verb)
            losses.update(dict(noun_lmd=self.LRW.lmd1))
            losses.update(dict(verb_lmd=self.LRW.lmd2))
            losses.update(dict(noun_verb_cls_loss=loss))
        else:
            losses.update(dict(noun_cls_loss=loss_score_noun))
            losses.update(dict(verb_cls_loss=loss_score_verb))

        return losses
    # def load_checkpoint(self):

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):

        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[3:])
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        # x = x.reshape((-1, num_seg) + x.shape[1:])
        # if self.with_segmental_consensus:
        #     x = self.segmental_consensus(x)
        #     x = x.squeeze(1)
        if self.with_cls_head:
            verb, noun = self.cls_head(x)
            noun = noun.reshape((-1, num_seg)+noun.shape[1:])
            verb = verb.reshape((-1, num_seg)+verb.shape[1:])
            noun, noun_final_logit = self.noun_segmental_consensus(None, noun)
            verb, verb_final_logit = self.verb_segmental_consensus(None, verb)
        if self.with_noun_head:
            noun = self.noun_cls_head(x)

        if self.with_verb_head:
            verb = self.verb_cls_head(x)

        return noun.cpu().numpy(), verb.cpu().numpy()
