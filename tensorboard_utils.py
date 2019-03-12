from collections import namedtuple
from operator import attrgetter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from os.path import join
import pandas as pd

summary_home = r'D:\tmp\fujian\temp_move2'
metric = namedtuple('metric', ['wall_time', 'step', 'acc', 'miou'])

res = []
for each_dir in os.listdir(summary_home):
    if not os.path.isdir(join(summary_home, each_dir)):
        continue
    print(each_dir)
    model, dataset, _, epochs, batch_size, init_lr, end_lr, iterations, crop_size, bn_scale, ignore_label, structure_model, extra_message = each_dir.split(
        '#')
    # front_end, back_end = model.split('_')
    # front_end, stride = front_end.split('@')

    cur_metrics = set()
    for each_summary_file in os.listdir(join(summary_home, each_dir, 'eval')):
        event_acc = EventAccumulator(join(summary_home, each_dir, 'eval', each_summary_file))
        event_acc.Reload()
        acc = event_acc.Scalars('accuracy')
        miou = event_acc.Scalars('mean_iou')

        for each_acc, each_miou in zip(acc, miou):
            cur_metrics.add(metric(each_acc.wall_time, each_acc.step, each_acc.value, each_miou.value))
    # print(cur_metrics)
    # exit(1)
    cur_metrics = sorted(cur_metrics, key=attrgetter('step'))
    run_time = cur_metrics[-1].wall_time - cur_metrics[0].wall_time

    best_acc = max(cur_metrics, key=attrgetter('acc'))
    best_miou = max(cur_metrics, key=attrgetter('miou'))
    cur = [model, init_lr, end_lr, batch_size, crop_size, epochs, iterations, run_time,
           cur_metrics[-1].acc, best_acc.acc, best_acc.step, cur_metrics[-1].miou, best_miou.miou, best_miou.step,
           bn_scale]
    res.append(cur)

res = pd.DataFrame(res, columns=[
    'model', 'init_lr', 'end_lr', 'batch_size', 'crop_size', 'epochs', 'iteration', 'runtime',
    'accuracy', 'best acc', 'best acc iter', 'mean iou', 'best miou', 'best miou iter', 'bn_scale'])
res.to_csv(join(summary_home, 'res.csv'), index=False)
