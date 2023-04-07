# -*- coding: utf-8 -*-
"""
@Time: 2023/4/7 10:21
@Auth: 除以七  ➗7️⃣
@File: tensorboard日志读取.py
@E-mail: divided.by.07@gmail.com
@Github: https://github.com/divided-by-7
@info: None
"""
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
# 加载日志：
logs_dir = r"part_det_new/logs-2023-04-06-19-21-05/Train_loss_loss_bbox_Train/events.out.tfevents.1680780264.hhd02-gpu.8662.3"
ea=event_accumulator.EventAccumulator(logs_dir)
ea.Reload()
print(ea.scalars.Keys())
loss_box=ea.scalars.Items('Train_loss/loss_bbox')
print(len(loss_box))
# print([(i.step,i.value) for i in loss_box])
data = [np.array((i.step,i.value)) for i in loss_box]

# 保存日志
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object
writer = SummaryWriter(log_dir='part_det_new/logs-2023-04-06-19-21-05/IoU_iter_curve')
for i in data:
    # writer.add_scalar('名字',y值,x值)
    writer.add_scalar('Train_IoU',1-i[1],i[0]) # 把loss值写入summary writer