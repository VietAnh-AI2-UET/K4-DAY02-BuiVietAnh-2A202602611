"""Hàm loss (đo sai số dự đoán) và trộn ảnh/nhãn cho bước 2."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("CE có trọng số cần weight tính từ tập train")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Không hỗ trợ loss: {kind}")


class LabelSmoothingCE(nn.Module):
    """Làm mềm nhãn: tránh yêu cầu xác suất lớp đúng phải sát 100%."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing phải thuộc [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Giảm đóng góp của ảnh dễ, để model chú ý hơn đến ảnh khó."""
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma không được âm")
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -(1 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Lớp ít ảnh nhận trọng số lớn hơn; chỉ dùng số ảnh TRAIN."""
    counts = torch.tensor(np.array(counts, copy=True), dtype=torch.float32)
    if (counts <= 0).any() or not 0 <= beta < 1:
        raise ValueError("Mỗi lớp cần có ảnh train, beta phải thuộc [0, 1)")
    weights = counts.reciprocal() if beta == 0 else (1 - beta) / (1 - beta ** counts)
    return weights / weights.mean()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn ảnh cùng nhãn; lam là phần đóng góp còn lại của ảnh gốc."""
    if alpha <= 0 or mode not in {"mixup", "cutmix"}:
        raise ValueError("alpha phải dương; mode là mixup hoặc cutmix")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(y), device=x.device)
    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        height, width = x.shape[-2:]
        ratio = np.sqrt(1 - lam)
        box_w, box_h = int(width * ratio), int(height * ratio)
        cx, cy = np.random.randint(width), np.random.randint(height)
        x1, x2 = max(0, cx - box_w // 2), min(width, cx + box_w // 2)
        y1, y2 = max(0, cy - box_h // 2), min(height, cy + box_h // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        # Tính lại theo hộp THỰC sau khi cắt biên, không dùng lam ban đầu.
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (height * width)
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
