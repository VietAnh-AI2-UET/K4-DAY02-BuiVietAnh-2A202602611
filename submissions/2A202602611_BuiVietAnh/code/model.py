"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    TODO:
      - timm.create_model(name, pretrained=..., num_classes=num_classes, drop_rate=...)
        (timm tự thay head mới; head khởi tạo ngẫu nhiên)
      - nếu init == "frozen": gọi freeze_backbone(model)
      - ghi lại tên tag trọng số thực sự được tải (model.pretrained_cfg)
    """
    import timm

    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Không hỗ trợ init: {init}")
    # timm giữ backbone (phần trích đặc trưng), thay head (lớp dự đoán) mới.
    model = timm.create_model(
        name, pretrained=pretrained and init != "scratch",
        num_classes=num_classes, drop_rate=drop_rate,
    )
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    TODO:
      - requires_grad = False cho tham số backbone; head (model.get_classifier()) vẫn train
      - lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm cũng phải ở chế độ eval.
        Hãy nghĩ nơi nào trong train loop phải gọi lại model.train() mà vẫn giữ BN ở eval.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad_(True)
    # Giữ thống kê BatchNorm và dropout ổn định khi chỉ học head.
    # Nếu gọi model.train() sau đó, vòng train phải đưa backbone về eval lại.
    model.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Tách backbone, norm/bias và head để đặt LR riêng."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = [[], [], []]
    for p in model.parameters():
        if p.requires_grad:
            groups[2 if id(p) in head_ids else (1 if p.ndim <= 1 else 0)].append(p)
    return [
        {"params": groups[0], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": groups[1], "lr": lr_backbone, "weight_decay": 0.0},
        {"params": groups[2], "lr": lr_head, "weight_decay": weight_decay},
    ]


def count_params(model) -> float:
    """Đếm toàn bộ tham số, đơn vị triệu (M)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """Đếm bằng fvcore: một phép nhân-cộng tính là một MAC.

    fvcore không đếm mọi phép toán; xem cảnh báo unsupported operators khi chạy.
    Đây là chi phí tính toán ước lượng, không phải thời gian chạy thực tế.
    """
    import torch
    from fvcore.nn import FlopCountAnalysis
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        sample = torch.zeros(1, 3, img_size, img_size, device=device)
        with torch.inference_mode():
            return float(FlopCountAnalysis(model, sample).total()) / 1e9
    finally:
        model.train(was_training)
