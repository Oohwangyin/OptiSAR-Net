from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可传入类别权重

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # 根据实例数量计算类别权重（逆频率）
        num_samples = [119265, 7570, 104042, 42161, 1031, 3893]
        total = sum(num_samples)
        class_weights = [total / (6 * c) for c in num_samples]
        self.class_weights = torch.tensor(class_weights, device=self.device)

    def criterion(self, preds, batch):
        # 替换原有的损失计算
        loss, loss_items = self.model(preds, batch)
        # 对 cls_loss 应用 focal loss
        if hasattr(self.model, 'criterion'):
            # 劫持 cls_loss 计算
            original_cls_loss = self.model.criterion.cls_loss
            self.model.criterion.cls_loss = lambda pred, target: FocalLoss(gamma=1.5, alpha=self.class_weights)(pred, target)
        return loss, loss_items