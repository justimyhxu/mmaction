import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                         mean_class_accuracy)
import os.path as osp

def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    parser.add_argument('--ignore_cache', action='store_true', help='whether to ignore cache')
    parser.add_argument('--challenge')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark

    cfg.work_dir = './work_dirs/' + cfg.filename.split('/')[-1].split('.')[0]
    if args.out is None:
        args.out = osp.join(cfg.work_dir,'results.pkl')
    if args.checkpoint is None:
        args.checkpoint = osp.join(cfg.work_dir, 'latest.pth')
        print('checkpoint_path', args.checkpoint)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if osp.exists(args.out) and not args.ignore_cache:
        outputs = mmcv.load(args.out)
    else:
        if args.gpus == 1:
            model = build_recognizer(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            load_checkpoint(model, args.checkpoint, strict=True)
            model = MMDataParallel(model, device_ids=[0])

            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                num_gpus=1,
                dist=False,
                shuffle=False)
            outputs = single_test(model, data_loader)
        else:
            model_args = cfg.model.copy()
            model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
            model_type = getattr(recognizers, model_args.pop('type'))
            outputs = parallel_test(
                model_type,
                model_args,
                args.checkpoint,
                dataset,
                _data_func,
                range(args.gpus),
                workers_per_gpu=args.proc_per_gpu)

    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

    gt_noun_labels = []
    gt_verb_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_noun_labels.append(ann['noun_label'])
        gt_verb_labels.append(ann['verb_label'])


    def top_acc(output, gt_labels, name):
        if args.use_softmax:
            print("Averaging score over {} clips with softmax".format(
                output[0].shape[0]))
            results = [softmax(res, dim=1).mean(axis=0) for res in output]
        else:
            print("Averaging score over {} clips without softmax (ie, raw)".format(
                output[0].shape[0]))
            results = [res.mean(axis=0) for res in output]

        top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
        # mean_acc = mean_class_accuracy(results, gt_labels)
        # print("{} Mean Class Accuracy = {:.02f}".format(name,mean_acc * 100))
        print("{} Top-1 Accuracy = {:.02f}".format(name,top1 * 100))
        print("{} Top-5 Accuracy = {:.02f}".format(name,top5 * 100))
    def convert_json(results):
        action_ids = [x.strip().split(',')[0] for x in open(cfg.data.test.ann_file)]
        assert len(action_ids) == len(results)
        final_dict = dict()
        final_dict.update(dict(
            version="0.1",
            challenge="action_recognition"
        ))
        results_dict = dict()
        for action_id,res in zip(action_ids, results):
            action_dict = {}
            noun, verb = results
            noun = noun.mean(axis=0)
            verb = verb.mean(axis=0)
            noun_re = dict(zip(map(str, list(range(len(noun)))),noun.to_list()))
            verb_re = dict(zip(map(str, list(range(len(verb)))),verb.to_list()))
            results_dict.update(action_id=dict(noun=noun_re,
                                verb=verb_re))
        final_dict.update(results=results_dict)
        mmcv.dump(final_dict, 'test.json')



    get_noun = lambda x:x[0]
    get_verb = lambda x:x[1]
    noun_output = list(map(get_noun,outputs))
    verb_output = list(map(get_verb, outputs))
    if args.challenge:

    else:
        top_acc(noun_output, gt_noun_labels,'noun')
        top_acc(verb_output, gt_verb_labels, 'verb')



if __name__ == '__main__':
    main()
