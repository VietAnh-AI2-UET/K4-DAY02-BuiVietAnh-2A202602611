"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    import os
    import random
    import numpy as np
    import torch

    # Cấu hình CUDA trước phép tính GPU để kết quả dễ tái lập hơn.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Báo lỗi nếu phép tính không có cách chạy xác định, tránh im lặng lệch kết quả.
    torch.use_deterministic_algorithms(True)


def build_optimizer(model, cfg: Config):
    """AdamW với LR backbone/head khác nhau, không decay norm/bias backbone."""
    import torch
    from model import param_groups
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Tăng LR trong warmup, rồi giảm theo cosine; cập nhật mỗi batch."""
    import math
    import torch
    total = cfg.epochs * steps_per_epoch
    warmup = min(int(cfg.warmup_epochs * steps_per_epoch), total)
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        raise NotImplementedError("TODO")

    def update(self, model) -> None:
        raise NotImplementedError("TODO")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Học một lượt qua train; thời gian không gồm đánh giá val."""
    import time
    import torch
    model.train()
    if cfg.init == "frozen":
        model.eval()
        model.get_classifier().train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    total_loss, count = 0.0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        # AMP dùng độ chính xác hỗn hợp để giảm bộ nhớ GPU.
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise ValueError("Loss train không hữu hạn")
        scaler.scale(loss).backward()
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # Nếu AMP bỏ cập nhật vì gradient quá lớn thì không bước lịch LR.
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        total_loss += loss.detach().item() * len(labels)
        count += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"train_loss": total_loss / count, "train_seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device):
    """Dự đoán đúng thứ tự loader, không cập nhật trọng số."""
    import torch
    import numpy as np
    model.eval()
    filenames, targets, outputs = [], [], []
    total_loss, count = 0.0, 0
    with torch.inference_mode():
        for images, labels, names in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * len(labels)
            count += len(labels)
            filenames.extend(names)
            targets.append(labels.cpu().numpy())
            outputs.append(logits.float().cpu().numpy())
    return filenames, np.concatenate(targets), np.concatenate(outputs), total_loss / count


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Lưu loss train/val và macro-F1 val để xem quá trình học."""
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("train_loss", "val_loss"):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_macro_f1"] for r in history])
    axes[1].set_ylabel("Macro-F1 val")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Chạy công thức nền bước 1; lưu checkpoint tốt nhất theo macro-F1 val."""
    import json
    import time
    from dataclasses import asdict
    import numpy as np
    import pandas as pd
    import torch
    import timm
    import dataset
    import model as model_utils
    from losses import build_criterion
    # Tìm repo từ vị trí train.py, không phụ thuộc thư mục mở notebook.
    import sys
    repo_dir = next((parent for parent in Path(__file__).resolve().parents
                     if (parent / "eval.py").is_file()), None)
    if repo_dir is None:
        raise FileNotFoundError("Không tìm thấy eval.py trong repo bài lab")
    sys.path.insert(0, str(repo_dir))
    from eval import compute_metrics, save_predictions

    # Những lựa chọn nâng cao còn TODO ở bước 2: báo rõ thay vì bỏ qua cấu hình.
    if cfg.mix is not None or cfg.ema_decay is not None or cfg.class_weight_beta is not None:
        raise NotImplementedError("Mix/EMA/class weights chưa cài đặt; bước 1 dùng công thức nền")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.warmup_epochs < 0:
        raise ValueError("epochs/batch_size phải dương; warmup_epochs không âm")
    set_seed(cfg.seed)
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    # Tránh vô tình ghi đè một lần chạy trước; dùng exp_id mới nếu muốn chạy lại.
    if (output / "config.json").exists():
        raise FileExistsError(f"{output} đã có lần chạy; đổi exp_id hoặc out_dir")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
        cfg.batch_size, False, num_workers=cfg.num_workers)
    if len(train_loader) == 0:
        raise ValueError("Không có batch train: giảm batch_size")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_utils.build_model(cfg.backbone, pretrained=cfg.init != "scratch",
        num_classes=dataset.NUM_CLASSES, drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    pretrained_cfg = model.pretrained_cfg
    tag = pretrained_cfg.get("tag", "")
    weight_tag = (pretrained_cfg.get("architecture", cfg.backbone.split(".")[0])
                  + (f".{tag}" if tag else "")) if cfg.init != "scratch" else "scratch"
    params_m = model_utils.count_params(model)
    gmac = model_utils.count_gmacs(model, cfg.img_size)
    # Đếm GMAC có thể chạy forward; seed lại trước quá trình học thật.
    set_seed(cfg.seed)
    metadata = {**asdict(cfg), "weight_tag": weight_tag, "torch_version": torch.__version__,
                "timm_version": timm.__version__, "device": str(device),
                "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                "gmac_tool": "fvcore (1 multiply-add = 1 MAC; unsupported ops excluded)"}
    (output / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history, best_f1, best_epoch = [], -1.0, 0
    checkpoint = output / "best.pt"
    for epoch in range(1, cfg.epochs + 1):
        row = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device)
        names, targets, logits, val_loss = evaluate(model, val_loader, criterion, device)
        probs = torch.from_numpy(logits).softmax(dim=1).numpy()
        metrics = compute_metrics(targets, probs.argmax(axis=1), probs)
        if not np.isfinite(val_loss) or not np.isfinite(logits).all():
            raise ValueError("Val loss/logits không hữu hạn")
        row.update(epoch=epoch, val_loss=val_loss, val_macro_f1=metrics["macro_f1"], val_top1=metrics["top1"])
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        # Dấu > giữ epoch sớm hơn nếu hai epoch có cùng điểm.
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save(model.state_dict(), checkpoint)
        print(f"{cfg.exp_id} epoch {epoch}/{cfg.epochs}: F1={metrics['macro_f1']:.4f}, "
              f"top1={metrics['top1']:.4f}, train={row['train_seconds']:.1f}s", flush=True)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    names, targets, logits, _ = evaluate(model, val_loader, criterion, device)
    probs = torch.from_numpy(logits).softmax(dim=1).numpy()
    metrics = compute_metrics(targets, probs.argmax(axis=1), probs)
    np.save(output / "val_logits.npy", logits)
    save_predictions(pred_path(cfg, "val"), names, targets, probs)
    # Test chỉ được dự đoán khi người dùng bật cờ ở vòng chung kết (bước 4).
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir,
            dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=cfg.num_workers)
        test_names, test_targets, test_logits, _ = evaluate(model, test_loader, criterion, device)
        np.save(output / "test_logits.npy", test_logits)
        save_predictions(pred_path(cfg, "test"), test_names, test_targets,
                         torch.from_numpy(test_logits).softmax(dim=1).numpy())

    # Độ trễ sơ bộ: batch 1, FP32, không tính đọc/chuẩn hoá ảnh.
    sample = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=device)
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png", f"{cfg.exp_id} - {cfg.backbone}")
    result = {"exp_id": cfg.exp_id, "backbone": cfg.backbone, "weight_tag": weight_tag,
              "params_M": params_m, "GMAC": gmac, "img_size": cfg.img_size,
              "epochs": cfg.epochs, "seed": cfg.seed, "best_epoch": best_epoch,
              "val_macro_f1": metrics["macro_f1"], "val_top1": metrics["top1"],
              "train_seconds_per_epoch": float(np.mean([r["train_seconds"] for r in history])),
              "latency_batch1_ms": latency_ms, "notes": "Latency: 1 measurement, FP32, excludes preprocessing"}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    raise NotImplementedError("TODO")


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
