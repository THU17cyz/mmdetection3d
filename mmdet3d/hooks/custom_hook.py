# Copyright (c) Open-MMLab. All rights reserved.
import torch
from mmcv.runner import HOOKS, master_only
from mmcv.runner.hooks import TensorboardLoggerHook
from mmcv.utils import TORCH_VERSION
from os import path as osp


@HOOKS.register_module()
class TensorboardHistogramLoggerHook(TensorboardLoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True):
        super(TensorboardHistogramLoggerHook,
              self).__init__(interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def after_run(self, runner):
        self.writer.close()

    @master_only
    def before_run(self, runner):
        if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        ratios = []
        vals = []
        rel_ratios = []
        rel_vals = []
        for tag, val in tags.items():
            if 'vote_ratio' in tag:
                t1, t2 = [float(x) for x in tag.split('_')[-1].split('-')]
                ratios.append(val)
                vals.append((t1 + t2) / 2)
            if 'vote_rel_ratio' in tag:
                t1, t2 = [float(x) for x in tag.split('_')[-1].split('-')]
                rel_ratios.append(val)
                rel_vals.append((t1 + t2) / 2)
            # if 'vote_rel_ratio' in tag or 'vote_ratio' in tag:
            #     continue
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif isinstance(val, float) or len(val.shape) == 0:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
            else:
                # print(val)
                # hist = tf.constant(val)
                # summary = tf.Summary(value=[tf.Summary.Value(
                # tag=tag, histo=hist)])
                # self.writer.add_summary(summary, self.get_iter(runner))
                self.writer.add_histogram(tag, val, self.get_iter(runner))

        if len(ratios) > 0:
            hist = []
            for ratio, val in zip(ratios, vals):
                hist.extend([val] * int(ratio * 10000))
            self.writer.add_histogram('vote_ratio', torch.tensor(hist),
                                      self.get_iter(runner))

        if len(rel_ratios) > 0:
            hist = []
            for ratio, val in zip(rel_ratios, rel_vals):
                hist.extend([val] * int(ratio * 10000))
            self.writer.add_histogram('vote_rel_ratio', torch.tensor(hist),
                                      self.get_iter(runner))
