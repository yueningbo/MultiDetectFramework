import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, max_lr, last_epoch=-1):
        # Warm-up 学习率从 0 增长到 max_lr
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        super().__init__(optimizer, self.get_lr_lambda, last_epoch)

    def get_lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            # Warm-up 阶段，学习率从 0 增长到 max_lr
            return float(epoch) / float(max(1, self.warmup_epochs))
        else:
            # 后续阶段使用预设的学习率
            return 1.0
