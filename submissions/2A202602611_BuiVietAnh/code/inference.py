"""Suy luận trên val/test: nhiều góc nhìn, gộp dự đoán và hiệu chỉnh xác suất."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    model.eval()
    names, targets, logits = [], [], []
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)
            if view is not None:
                images = view(images)
            output = model(images)
            names.extend(filenames)
            targets.append(labels.numpy())
            logits.append(output.float().cpu().numpy())
    return names, np.concatenate(targets), np.concatenate(logits)


def view_identity(x):
    return x


def view_hflip(x):
    # Chiều cuối là chiều rộng: đảo chiều này để lật ngang.
    return x.flip(-1)


def views_multicrop(x, crop: int):
    """4 góc và giữa; cả 5 ảnh đều có kích thước model đã học."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError("crop không được lớn hơn ảnh đầu vào")
    return [x[..., y:y + crop, z:z + crop] for y, z in
            [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop),
             ((h - crop) // 2, (w - crop) // 2)]]


def views_multiscale(x, sizes):
    # Chỉ dùng với model chấp nhận nhiều kích thước (CNN); ViT/Swin cần xử lý riêng.
    return [F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) for size in sizes]


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Nhiệt độ T phải dương và hữu hạn")
    return torch.as_tensor(logits, dtype=torch.float64).div(T).softmax(-1).numpy()


def aggregate_views(logits_per_view, space: str = "prob"):
    if not logits_per_view:
        raise ValueError("Cần ít nhất một view")
    if space == "prob":
        return np.mean([apply_temperature(x, 1.0) for x in logits_per_view], axis=0)
    if space == "logit":
        return apply_temperature(np.mean(logits_per_view, axis=0), 1.0)
    raise ValueError("space phải là prob hoặc logit")


def ensemble_probs(list_of_probs):
    if not list_of_probs:
        raise ValueError("Cần ít nhất một model")
    probs = np.mean(list_of_probs, axis=0)
    return probs / probs.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T bằng lưới thô rồi tinh, chỉ trên VAL; không thay đổi lớp argmax."""
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if not torch.isfinite(logits).all():
        raise ValueError("Logits không hữu hạn")
    def search(grid):
        # Tính trên CPU, không chiếm GPU đang huấn luyện.
        losses = [F.cross_entropy(logits / float(t), labels).item() for t in grid]
        return float(grid[int(np.argmin(losses))])
    coarse = search(np.geomspace(0.05, 20.0, 61))
    return search(np.geomspace(max(0.05, coarse / 1.2), min(20.0, coarse * 1.2), 41))


def fuse_conv_bn(model):
    """Gộp các cặp chắc chắn nối tiếp nhau; trả về bản sao để giữ model gốc."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    result = deepcopy(model).eval()
    def visit(module):
        for child in module.children():
            visit(child)
        if isinstance(module, nn.Sequential):
            children = list(module.named_children())
            pairs = [(children[i], children[i + 1]) for i in range(len(children) - 1)]
        else:
            # Các cặp quy ước của ResNet. Không gộp bừa mọi module nằm cạnh nhau.
            pairs = [((c, getattr(module, c)), (b, getattr(module, b)))
                     for c, b in [("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")]
                     if hasattr(module, c) and hasattr(module, b)]
        for (conv_name, conv), (bn_name, bn) in pairs:
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
    visit(result)
    return result


def predict_batch(model, images, method="identity", crop=224, temperature=1.0):
    """Trả về xác suất trên GPU; dùng chung lúc dự đoán và đo độ trễ."""
    if method == "fivecrop":
        views = views_multicrop(images, crop)
    elif method in {"hflip_prob", "hflip_logit"}:
        views = [images, view_hflip(images)]
    elif method == "identity":
        views = [images]
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")
    outputs = [model(view).float() for view in views]
    if method == "hflip_logit":
        return torch.stack(outputs).mean(0).div(temperature).softmax(-1)
    probs = torch.stack([out.softmax(-1) for out in outputs]).mean(0)
    # Dùng log(prob) làm điểm số chung cho cả 1-view và TTA rồi chia T.
    # T=1 trả lại xác suất gốc; lớp dự đoán được giữ nguyên khi hiệu chỉnh.
    return probs if temperature == 1.0 else probs.clamp_min(1e-12).log().div(temperature).softmax(-1)


def predict_probs(model, loader, device, method="identity", crop=224, temperature=1.0):
    model.eval()
    names, labels, probabilities = [], [], []
    with torch.inference_mode():
        for images, targets, filenames in loader:
            probs = predict_batch(model, images.to(device), method, crop, temperature)
            names.extend(filenames)
            labels.append(targets.numpy())
            probabilities.append(probs.cpu().numpy())
    return names, np.concatenate(labels), np.concatenate(probabilities)
