"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    import time
    import numpy as np
    if warmup < 10 or iters < 50:
        raise ValueError("Cần warmup >= 10 và iters >= 50")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    elapsed = []
    for _ in range(iters):
        sync()
        started = time.perf_counter()
        fn()
        sync()
        elapsed.append((time.perf_counter() - started) * 1000)
    return {"p50": float(np.percentile(elapsed, 50)), "p95": float(np.percentile(elapsed, 95)),
            "p99": float(np.percentile(elapsed, 99)), "mean": float(np.mean(elapsed)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    import torch
    from copy import deepcopy
    device = torch.device(device)
    if dtype not in {"fp32", "amp", "fp16"} or (device.type != "cuda" and dtype != "fp32"):
        raise ValueError("CPU chỉ đo FP32; GPU dùng fp32, amp hoặc fp16")
    measured_model = deepcopy(model).to(device).eval()
    if dtype == "fp16":
        measured_model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    def forward():
        with torch.autocast(device_type=device.type, enabled=dtype == "amp"):
            return measured_model(x)
    with torch.inference_mode():
        stats = bench(forward, warmup, iters,
                      sync=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None)
    return {**stats, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "images_per_s": batch_size / (stats["p50"] / 1000), "torch": torch.__version__,
            "preprocessing": "excluded", "fused_bn": False}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    import torch
    if k_views < 1:
        raise ValueError("k_views phải dương")
    class RepeatedViews(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            # Đo thật K lượt, có lật xen kẽ và gộp xác suất, không nhân số đo 1-view.
            return torch.stack([self.inner(x if i % 2 == 0 else x.flip(-1)).softmax(-1)
                                for i in range(k_views)]).mean(0)
    return {**latency_report(RepeatedViews(model), **kw), "k_views": k_views}
