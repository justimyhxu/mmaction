import mmcv
import numpy as np
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import random
import argparse

import cv2
import torch
import numpy as np
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

_TIMESTAMP_BIAS = 600
_TIMESTAMP_START = 840  # 60*14min
_TIMESTAMP_END = 1860  # 60*31min
_FPS = 30


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

cfg.merge_from_file(args.config)
cfg.CUDA = torch.cuda.is_available()
device = torch.device('cuda' if cfg.CUDA else 'cpu')
# random.seed(1)
# Firstly Load Proposal

class Tracking_Proposal(object):
    def __init__(self,
                 img_prefix,
                 proposal_path,
                 video_stat_file,
                 new_length=32,
                 new_step=2,
                 with_pysot=True):
        self.img_prefix = img_prefix
        self.new_step = new_step
        self.new_length = new_length
        self.proposal_dict = self.load_proposal(proposal_path)
        self.video_stats = dict([tuple(x.strip().split(' ')) for x in open(video_stat_file)])
        self.with_model = False
        self.with_pysot = with_pysot

    def load_proposal(self, path):
        proposal_dict = mmcv.load(path)
        convert_dict = {}
        for key, value in proposal_dict.items():
            video_id, frame = key.split(',')
            if convert_dict.get(video_id,None) is None:
                convert_dict[video_id] = {}
            elif convert_dict[video_id].get(frame, None) is None:
                convert_dict[frame] = {}
            convert_dict[video_id][frame] = value
        return convert_dict

    def _load_image(self, directory, image_tmpl, modality, idx):
        if modality in ['RGB', 'RGBDiff']:
            return mmcv.imread(osp.join(directory, image_tmpl.format(idx)))
        elif modality == 'Flow':
            x_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('x', idx)),
                flag='grayscale')
            y_imgs = mmcv.imread(
                osp.join(directory, image_tmpl.format('y', idx)),
                flag='grayscale')
            return [x_imgs, y_imgs]
        else:
            raise ValueError(
                'Not implemented yet; modality should be '
                '["RGB", "RGBDiff", "Flow"]')

    def tracking(self):
        keys = list(self.proposal_dict.keys())
        # random.shuffle(keys)
        shuffle_dict = [(key, self.proposal_dict[key]) for key in keys]

        for video_id, frame_info in shuffle_dict:
            for timestamp, proposals in frame_info.items():
                indice = _FPS * (int(timestamp) - _TIMESTAMP_START) + 1
                image_tmpl = 'img_{:05}.jpg'
                # forward tracking
                print('video_id:{}, frame:{}'.format(video_id, timestamp))
                for proposal in proposals:
                    width, height  = [int(ll) for ll in self.video_stats[video_id].split('x')]
                    ROI = np.array([int(x) for x in  (proposal * np.array([
                        width, height, width, height, 1
                    ]))[:4]])
                    track_window = tuple(np.concatenate([ROI[:2],ROI[-2:]-ROI[:2]],axis=0).tolist())

                    ann_frame = self._load_image(osp.join(self.img_prefix,
                                                          video_id),
                                                          image_tmpl, 'RGB', indice)
                    if True:
                        plt.imshow(ann_frame[:,:,::-1])
                        color = (random.random(), random.random(), random.random())
                        rect = plt.Rectangle((track_window[0],track_window[1]),
                                             track_window[2],
                                             track_window[3], fill=False,
                                             edgecolor=color, linewidth=3.5)
                        plt.gca().add_patch(rect)
                        plt.show()
                    # Forcasting Tracking
                    p = indice - self.new_step
                    for i, ind in enumerate(
                            range(-2, -(self.new_length+1), -self.new_step)):
                        unann_frame = self._load_image(osp.join(self.img_prefix,
                                                                video_id),
                                                                image_tmpl, 'RGB', p)
                        if self.with_pysot:
                            track_window = self.pysot_tracking_roi(track_window,
                                                                      org_frame=ann_frame,
                                                                      tracked_frame=unann_frame)
                        else:
                            track_window = self.cv2_tracking_roi(track_window,
                                                                  org_frame=ann_frame,
                                                                  tracked_frame=unann_frame)
                        ann_frame = unann_frame
                        p -= self.new_step

                    track_window = tuple(np.concatenate([ROI[:2], ROI[-2:] - ROI[:2]], axis=0).tolist())
                    ann_frame = self._load_image(osp.join(self.img_prefix,
                                                          video_id),
                                                          image_tmpl, 'RGB', indice)
                    # Backcasting Tracking
                    p = indice + self.new_step
                    for i, ind in enumerate(
                            range(0, self.new_length-1, self.new_step)):
                        unann_frame = self._load_image(osp.join(self.img_prefix,
                                                                video_id),
                                                                image_tmpl, 'RGB', p)
                        if self.with_pysot:
                            track_window = self.pysot_tracking_roi(track_window,
                                                                   org_frame=ann_frame,
                                                                   tracked_frame=unann_frame)
                        else:
                            track_window = self.cv2_tracking_roi(track_window,
                                                                 org_frame=ann_frame,
                                                                 tracked_frame=unann_frame)
                        ann_frame = unann_frame
                        p += self.new_step

    def build_model(self):
        model = ModelBuilder()
        # load model
        model.load_state_dict(torch.load(args.snapshot,
                                         map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)
        # build tracker
        tracker = build_tracker(model)
        return tracker

    def init_tracker(self, track_window, frame):
        self.tracking_model = self.build_model()
        self.tracking_model.init(frame, track_window)

    def pysot_tracking_roi(self, track_window, key_frame, tracked_frame, vis=True):
        if not self.with_model:
            self.init_tracker(track_window, key_frame)
            self.with_model = True

        outputs = self.tracker.track(tracked_frame)
        if 'polygon' in outputs:
            polygon = np.array(outputs['polygon']).astype(np.int32)
            cv2.polylines(tracked_frame, [polygon.reshape((-1, 1, 2))],
                          True, (0, 255, 0), 3)
            mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
            mask = mask.astype(np.uint8)
            mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
            tracked_frame = cv2.addWeighted(tracked_frame, 0.77, mask, 0.23, -1)
        else:
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(tracked_frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 3)

        return _

    def cv2_tracking_roi(self, track_window, org_frame, tracked_frame, vis=True):
        x, y, w, h = track_window
        roi = org_frame[y:y+h, x:x+w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv2.cvtColor(tracked_frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        if vis:
            # Draw it on image
            # pts = cv2.boxPoints(ret)
            # pts = np.int0(pts)
            # img2 = cv2.polylines(tracked_frame, [pts], True, 255, 2)
            x, y, w, h = track_window
            img2 = cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), 255, 2)
            plt.imshow(img2[:,:,::-1])
            plt.show()
        return track_window


data_root = 'data/ava/rawframes/'
tracking_inst = Tracking_Proposal(
                img_prefix=data_root,
                proposal_path='data/ava/ava_dense_proposals_train.FAIR.recall_93.9.pkl',
                video_stat_file='data/ava/ava_video_resolution_stats.csv',
                new_length=2,
                new_step=2
                )
tracking_inst.tracking()
