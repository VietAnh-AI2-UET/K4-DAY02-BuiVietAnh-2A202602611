"""Các bước 2–4 dùng chung train.run; notebook chỉ điều khiển thứ tự chạy.

T00 là công thức nền trên backbone đã chọn ở bước 1. B01 vẫn là mốc ResNet
trong bảng Backbones. Mọi lựa chọn được chốt bằng VAL trước khi mở TEST.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms

import dataset
import model as model_utils
import train
import inference
from benchmark import bench

REPO = next(p for p in Path(__file__).resolve().parents if (p / "eval.py").is_file())
sys.path.insert(0, str(REPO))
from eval import compute_metrics, save_predictions


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ghi file tạm rồi thay tên để tránh file JSON bị viết dở khi phiên ngắt.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_sheet(frame, sheet, workbook="results.xlsx"):
    path = Path(workbook)
    options = {"mode": "a", "if_sheet_exists": "replace"} if path.exists() else {"mode": "w"}
    # Đóng results.xlsx trong Excel trước khi chạy, để Windows cho phép ghi file.
    with pd.ExcelWriter(path, engine="openpyxl", **options) as writer:
        frame.to_excel(writer, sheet_name=sheet, index=False)
        writer.sheets[sheet].freeze_panes = "A2"
    frame.to_csv(path.with_name(f"{sheet.lower()}.csv"), index=False)


def load_config(folder):
    data = read_json(Path(folder) / "config.json")
    return train.Config(**{field.name: data[field.name] for field in fields(train.Config)})


def run_cached(cfg):
    """Đọc lần chạy đã hoàn tất; không tự ghi đè lần chạy đang dở."""
    folder = train.run_dir(cfg)
    if (folder / "summary.json").exists():
        data = read_json(folder / "config.json")
        differences = [key for key, value in asdict(cfg).items() if data.get(key) != value]
        if differences:
            raise ValueError(f"{folder}: cấu hình khác {differences}; dùng thư mục runs mới")
        if data.get("torch_version") != torch.__version__ or data.get("timm_version") != timm.__version__:
            raise ValueError(f"{folder}: phiên bản torch/timm khác lần trước")
        if not (folder / "best.pt").exists():
            raise FileNotFoundError(f"Thiếu checkpoint: {folder / 'best.pt'}")
        print(f"Đọc lần chạy đã có: {cfg.exp_id}, seed={cfg.seed}")
        return read_json(folder / "summary.json")
    return train.run(cfg)


def training_step(images_dir, labels_dir, out_dir="runs", pred_dir="predictions", workbook="results.xlsx"):
    """Một backbone; mỗi thí nghiệm chỉ đổi một nhóm yếu tố so với T00."""
    selection = read_json("backbone_selection.json")
    backbones = pd.read_csv("backbones.csv")
    chosen = selection["selected_backbones"][0]
    row = backbones.loc[backbones.backbone.eq(chosen)].iloc[0]
    original_folder = Path(out_dir) / row.exp_id / f"seed{int(row.seed)}"
    original = load_config(original_folder)
    # Ghim đúng tag trọng số bước 1; không để timm tự đổi tag mặc định.
    weight_tag = read_json(original_folder / "config.json")["weight_tag"]
    baseline = replace(original, exp_id="T00", backbone=weight_tag, seed=0,
                       images_dir=str(images_dir), labels_dir=str(labels_dir),
                       out_dir=str(out_dir), pred_dir=str(pred_dir), save_test_predictions=False)
    recipes = [
        ("T01", "B", "Thêm đổi màu ảnh", {"aug": "color"}),
        ("T02", "B", "RandAugment: biến đổi ảnh ngẫu nhiên", {"aug": "randaug"}),
        ("T03", "C", "Làm mềm nhãn 0.1", {"loss": "ls", "label_smoothing": 0.1}),
        ("T04", "C", "Focal: chú ý ảnh khó, gamma=2", {"loss": "focal", "focal_gamma": 2.0}),
        ("T05", "C", "CE có trọng số nghịch số ảnh train", {"loss": "ce_weighted"}),
        ("T06", "D", "Lấy mẫu cân bằng lớp", {"sampler": "balanced"}),
        ("T07", "F", "EMA: làm mượt trọng số, decay=0.99", {"ema_decay": 0.99}),
        ("T08", "B", "CutMix: trộn vùng ảnh cùng nhãn", {"mix": "cutmix", "mix_alpha": 1.0}),
    ]
    rows = []
    base_result = run_cached(baseline)
    baseline_f1 = base_result["val_macro_f1"]
    rows.append({**base_result, "axis": "baseline", "changed": "Công thức nền trên backbone đã chọn",
                 "delta_vs_T00": 0.0})
    save_sheet(pd.DataFrame(rows), "Training", workbook)
    results = {}
    for exp_id, axis, description, changes in recipes:
        cfg = replace(baseline, exp_id=exp_id, **changes)
        result = run_cached(cfg)
        results[exp_id] = result
        # Bổ sung F1 hai lớp khó từ đúng file dự đoán đã lưu, không dự đoán lại.
        predictions = pd.read_csv(train.pred_path(cfg, "val"))
        probs = predictions[[f"p{i}" for i in range(dataset.NUM_CLASSES)]].to_numpy()
        metrics = compute_metrics(predictions.y_true.to_numpy(), probs.argmax(1), probs)
        rows.append({**result, "axis": axis, "changed": description,
                     "delta_vs_T00": result["val_macro_f1"] - baseline_f1,
                     "f1_chinee_apple": float(metrics["f1"][0]), "f1_snake_weed": float(metrics["f1"][7])})
        save_sheet(pd.DataFrame(rows), "Training", workbook)

    # Lấy lựa chọn tốt nhất của từng trục rồi kết hợp các trục cải thiện F1.
    best_per_axis = {}
    for exp_id, axis, _, changes in recipes:
        candidate = (results[exp_id]["val_macro_f1"], exp_id, changes)
        if axis not in best_per_axis or candidate[0] > best_per_axis[axis][0]:
            best_per_axis[axis] = candidate
    ranked_axes = sorted(best_per_axis.values(), key=lambda item: item[0], reverse=True)
    combined = [item for item in ranked_axes if item[0] > baseline_f1]
    # Nếu ít hơn 2 trục cải thiện, vẫn thử 2 lựa chọn tốt nhất và ghi là thăm dò.
    exploratory = len(combined) < 2
    if exploratory:
        combined = ranked_axes[:2]
    merged = {}
    for _, _, changes in combined:
        merged.update(changes)
    combo_cfg = replace(baseline, exp_id="T09", **merged)
    combo_result = run_cached(combo_cfg)
    rows.append({**combo_result, "axis": "combination", "changed": "+".join(item[1] for item in combined),
                 "delta_vs_T00": combo_result["val_macro_f1"] - baseline_f1,
                 "notes": "Kết hợp thăm dò; chưa có 2 trục thắng nền" if exploratory else "Kết hợp các trục cải thiện"})
    table = pd.DataFrame(rows)
    save_sheet(table, "Training", workbook)
    best = table.sort_values(["val_macro_f1", "train_seconds_per_epoch"], ascending=[False, True]).iloc[0]
    best_folder = Path(out_dir) / best.exp_id / "seed0"
    write_json("training_selection.json", {"baseline_config": asdict(baseline),
               "best_config": asdict(load_config(best_folder)), "best_exp_id": best.exp_id,
               "val_macro_f1": float(best.val_macro_f1), "delta_vs_T00": float(best.delta_vs_T00),
               "note": "Một seed: Δ là quan sát sơ bộ; so với std ở bước 4, chưa kết luận chắc chắn."})
    print(f"Chọn {best.exp_id}: F1={best.val_macro_f1:.4f}; Δ={best.delta_vs_T00:+.4f} so với T00")
    print("Kết hợp T09 cải thiện so với T00:", f"{combo_result['val_macro_f1'] - baseline_f1:+.4f}")
    return table


def make_eval_loader(cfg, frame, method):
    if method == "fivecrop":
        # Giữ ảnh 256 để cắt 5 vùng 224; không cắt giữa trước rồi cắt lại.
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                        transforms.Normalize(dataset.IMAGENET_MEAN, dataset.IMAGENET_STD)])
    else:
        transform = dataset.build_transforms(False, cfg.img_size)
    return dataset.make_loader(frame, cfg.images_dir, transform, cfg.batch_size,
                               False, num_workers=cfg.num_workers)


def load_checkpoint_model(cfg, device):
    # Nạp checkpoint nên không tải lại trọng số pretrained từ Internet.
    model = model_utils.build_model(cfg.backbone, pretrained=False, num_classes=dataset.NUM_CLASSES,
                                    drop_rate=cfg.drop_rate, init="finetune").to(device).eval()
    model.load_state_dict(torch.load(train.run_dir(cfg) / "best.pt", map_location=device, weights_only=True))
    return model


def measure_method(model, cfg, device, method, temperature=1.0, batches=(1, 32), iters=50):
    reports = []
    size = 256 if method == "fivecrop" else cfg.img_size
    for batch in batches:
        sample = torch.zeros(batch, 3, size, size, device=device)
        def forward():
            return inference.predict_batch(model, sample, method, cfg.img_size, temperature)
        with torch.inference_mode():
            stats = bench(forward, warmup=10, iters=iters,
                          sync=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None)
        reports.append({**stats, "batch": batch, "images_per_s": batch * 1000 / stats["p50"],
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                        "dtype": "fp32", "img_size": cfg.img_size, "input_size": size,
                        "fused_bn": False, "torch": torch.__version__, "method": method,
                        "preprocessing": "Không tính đọc/resize/normalize; có tính tạo view và gộp xác suất"})
    return reports


def inference_step(workbook="results.xlsx", iters=50, batches=(1, 32)):
    """Bốn cách ngoài I00; không huấn luyện và không mở tập test."""
    cfg = train.Config(**read_json("training_selection.json")["best_config"])
    # Chỉ đọc CSV VAL. Không dùng load_split ở bước chọn cách suy luận.
    val_df = pd.read_csv(Path(cfg.labels_dir) / f"val_subset{cfg.fold}.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(cfg, device)
    specifications = [("I00", "identity", "1 view - mốc", 1),
                      ("I01", "hflip_prob", "Lật ngang, gộp xác suất", 2),
                      ("I02", "fivecrop", "5 crop, gộp xác suất", 5),
                      ("I03", "hflip_logit", "Lật ngang, gộp điểm số trước softmax", 2),
                      ("I04", "identity", "Hiệu chỉnh xác suất bằng nhiệt độ T", 1)]
    rows, latency_rows, uncals = [], [], {}
    val_names, val_labels = None, None
    for exp_id, method, description, views in specifications:
        if method not in uncals:
            names, targets, probs = inference.predict_probs(model, make_eval_loader(cfg, val_df, method),
                                                            device, method, cfg.img_size)
            if val_names is not None and (names != val_names or not np.array_equal(targets, val_labels)):
                raise ValueError("Thứ tự file/nhãn giữa các phương pháp khác nhau")
            val_names, val_labels = names, targets
            uncals[method] = probs
        uncal = uncals[method]
        temperature = inference.fit_temperature(np.log(np.clip(uncal, 1e-12, 1)), val_labels) if exp_id == "I04" else 1.0
        probs = inference.apply_temperature(np.log(np.clip(uncal, 1e-12, 1)), temperature) if exp_id == "I04" else uncal
        metrics = compute_metrics(val_labels, probs.argmax(1), probs)
        reports = measure_method(model, cfg, device, method, temperature, batches, iters)
        for report in reports:
            latency_rows.append({**report, "exp_id": exp_id, "checkpoint": str(train.run_dir(cfg) / "best.pt")})
        batch1 = next(report for report in reports if report["batch"] == 1)
        throughput = reports[-1]
        rows.append({"exp_id": exp_id, "method": method, "description": description,
                     "checkpoint": str(train.run_dir(cfg) / "best.pt"), "K": views,
                     "val_macro_f1": metrics["macro_f1"], "val_top1": metrics["top1"],
                     "val_ece": metrics["ece"], "val_nll": metrics["nll"], "temperature": temperature,
                     "p50_ms": batch1["p50"], "p95_ms": batch1["p95"], "p99_ms": batch1["p99"],
                     "images_per_s": throughput["images_per_s"], "throughput_batch": throughput["batch"],
                     "relative_cost": batch1["p50"] / rows[0]["p50_ms"] if rows else 1.0})
        save_predictions(Path(cfg.pred_dir) / f"{exp_id}_seed{cfg.seed}_val.csv", val_names, val_labels, probs)
        save_sheet(pd.DataFrame(rows), "Inference", workbook)
        save_sheet(pd.DataFrame(latency_rows), "Latency", workbook)
        print(f"{exp_id}: F1={metrics['macro_f1']:.4f}, ECE={metrics['ece']:.4f}, p95={batch1['p95']:.2f} ms")
    table = pd.DataFrame(rows)
    # Temperature không đổi lớp dự đoán: chọn kiểu view theo F1/độ trễ trước.
    best = table.loc[table.exp_id.ne("I04")].sort_values(["val_macro_f1", "p95_ms"], ascending=[False, True]).iloc[0]
    # Chốt dùng calibration; mỗi seed vòng cuối sẽ khớp lại T trên VAL của seed đó.
    write_json("inference_selection.json", {"method": best.method, "source_id": best.exp_id,
               "calibrate": True, "training_config": asdict(cfg), "val_macro_f1": float(best.val_macro_f1),
               "note": "T khớp trên val; ECE/NLL val sau khớp là số trong mẫu, test chỉ đánh giá ở bước 4."})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(table.p95_ms, table.val_macro_f1)
    for row in table.itertuples():
        ax.annotate(row.exp_id, (row.p95_ms, row.val_macro_f1))
    ax.set(xlabel="Độ trễ p95 batch 1 (ms)", ylabel="Macro-F1 val", title="Đánh đổi chất lượng và độ trễ")
    fig.tight_layout()
    Path("curves").mkdir(exist_ok=True)
    fig.savefig("curves/inference_tradeoff.png", dpi=150)
    plt.close(fig)
    print("Chọn cách suy luận:", best.method, "; sẽ khớp T trên val cho từng seed.")
    print("ECE 1-view trước/sau:", f"{rows[0]['val_ece']:.4f} / {rows[-1]['val_ece']:.4f}")
    return table


def final_prediction(cfg, method, device):
    """Một lượt qua test cho mỗi cấu hình/seed; lưu cả trước và sau calibration."""
    folder = train.run_dir(cfg)
    cache = folder / "final_outputs.npz"
    metadata_path = folder / "final_inference.json"
    spec = {"config": asdict(cfg), "method": method, "calibrate": cfg.exp_id == "F01"}
    if cache.exists():
        metadata = read_json(metadata_path)
        if metadata["spec"] != spec:
            raise ValueError("Đã đánh giá test với cấu hình khác; không được đổi cấu hình")
        with np.load(cache, allow_pickle=False) as archive:
            outputs = {key: archive[key] for key in archive.files}
        return outputs, metadata
    marker = folder / "test_started.json"
    if marker.exists():
        raise RuntimeError(f"{folder}: test đã bắt đầu nhưng chưa lưu xong. Dừng để tránh chạy test lần nữa; ghi sự cố vào báo cáo.")
    model = load_checkpoint_model(cfg, device)
    val_df = pd.read_csv(Path(cfg.labels_dir) / f"val_subset{cfg.fold}.csv")
    val_names, val_labels, val_uncal = inference.predict_probs(
        model, make_eval_loader(cfg, val_df, method), device, method, cfg.img_size)
    temperature = inference.fit_temperature(np.log(np.clip(val_uncal, 1e-12, 1)), val_labels) if spec["calibrate"] else 1.0
    val_probs = inference.apply_temperature(np.log(np.clip(val_uncal, 1e-12, 1)), temperature)
    latency = measure_method(model, cfg, device, method, temperature, batches=(1,), iters=50)[0]
    metadata = {"spec": spec, "temperature": temperature, "latency": latency}
    write_json(metadata_path, metadata)
    # Đánh dấu TRƯỚC khi dự đoán test. mode 'x' từ chối nếu đã có lượt chạy khác.
    with marker.open("x", encoding="utf-8") as handle:
        json.dump(spec, handle, ensure_ascii=False, indent=2)
    test_df = pd.read_csv(Path(cfg.labels_dir) / f"test_subset{cfg.fold}.csv")
    test_names, test_labels, test_uncal = inference.predict_probs(
        model, make_eval_loader(cfg, test_df, method), device, method, cfg.img_size)
    # Không forward thêm: calibration dùng lại xác suất của đúng một lượt test.
    test_probs = inference.apply_temperature(np.log(np.clip(test_uncal, 1e-12, 1)), temperature)
    outputs = {"val_names": np.asarray(val_names), "val_labels": val_labels, "val_probs": val_probs,
               "test_names": np.asarray(test_names), "test_labels": test_labels,
               "test_probs": test_probs, "test_uncal": test_uncal}
    temporary = folder / "final_outputs.tmp.npz"
    np.savez_compressed(temporary, **outputs)
    temporary.replace(cache)
    return outputs, metadata


def final_step(seeds=(0, 1, 2), workbook="results.xlsx"):
    """Chốt cấu hình, train cả chung kết và T00, tổng hợp mean/std qua seed."""
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Chung kết cần ít nhất 3 seed khác nhau")
    training_selection = read_json("training_selection.json")
    inference_selection = read_json("inference_selection.json")
    if inference_selection["training_config"] != training_selection["best_config"]:
        raise ValueError("Cấu hình huấn luyện đã đổi: phải làm lại bước 3 trước khi test")
    final_cfg = train.Config(**training_selection["best_config"])
    baseline_cfg = train.Config(**training_selection["baseline_config"])
    lock = {"final_config": asdict(final_cfg), "baseline_config": asdict(baseline_cfg),
            "method": inference_selection["method"], "calibrate": True, "seeds": list(seeds)}
    lock_path = Path(final_cfg.out_dir) / "final_selection_lock.json"
    if lock_path.exists():
        if read_json(lock_path) != lock:
            raise ValueError("Cấu hình chung kết đã chốt. Không đổi cấu hình sau khi mở test.")
    else:
        write_json(lock_path, lock)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, class_rows, matrices, latencies = [], [], [], []
    for seed in seeds:
        for exp_id, template, method in [("F01", final_cfg, lock["method"]), ("T00", baseline_cfg, "identity")]:
            # train.run chỉ đánh giá VAL. TEST chạy ở final_prediction để dùng đúng TTA đã chốt.
            cfg = replace(template, exp_id=exp_id, seed=seed, save_test_predictions=False)
            summary = run_cached(cfg)
            outputs, metadata = final_prediction(cfg, method, device)
            val_metrics = compute_metrics(outputs["val_labels"], outputs["val_probs"].argmax(1), outputs["val_probs"])
            test_metrics = compute_metrics(outputs["test_labels"], outputs["test_probs"].argmax(1), outputs["test_probs"])
            uncal_metrics = compute_metrics(outputs["test_labels"], outputs["test_uncal"].argmax(1), outputs["test_uncal"])
            save_predictions(train.pred_path(cfg, "val"), outputs["val_names"], outputs["val_labels"], outputs["val_probs"])
            save_predictions(train.pred_path(cfg, "test"), outputs["test_names"], outputs["test_labels"], outputs["test_probs"])
            if exp_id == "F01":
                save_predictions(Path(cfg.pred_dir) / f"F01uncal_seed{seed}_test.csv",
                                 outputs["test_names"], outputs["test_labels"], outputs["test_uncal"])
            latency = metadata["latency"]
            latencies.append({**latency, "exp_id": exp_id, "seed": seed})
            rows.append({**summary, "kind": "seed", "method": method, "temperature": metadata["temperature"],
                         "val_macro_f1": val_metrics["macro_f1"], "val_top1": val_metrics["top1"],
                         "test_macro_f1": test_metrics["macro_f1"], "test_top1": test_metrics["top1"],
                         "test_ece": test_metrics["ece"], "test_ece_uncal": uncal_metrics["ece"],
                         "val_test_f1_gap": abs(val_metrics["macro_f1"] - test_metrics["macro_f1"]),
                         "p95_ms": latency["p95"]})
            matrices.append((exp_id, seed, test_metrics["confusion"]))
            for label, name in enumerate(dataset.CLASS_NAMES):
                class_rows.append({"exp_id": exp_id, "seed": seed, "class": name,
                                   "support": int(test_metrics["support"][label]),
                                   **{key: float(test_metrics[key][label]) for key in ("precision", "recall", "f1")}})
            # Lưu sau từng seed/cấu hình; nếu ô chạy lại sẽ chỉ đọc cache test.
            save_sheet(pd.DataFrame(rows), "Final", workbook)
            save_sheet(pd.DataFrame(class_rows), "PerClass", workbook)
            print(f"{exp_id} seed={seed}: test F1={test_metrics['macro_f1']:.4f}, top1={test_metrics['top1']:.4f}")
    table = pd.DataFrame(rows)
    metric_cols = ["val_macro_f1", "val_top1", "test_macro_f1", "test_top1", "test_ece", "test_ece_uncal", "p95_ms"]
    aggregated = []
    for exp_id, group in table.groupby("exp_id"):
        aggregate = {"exp_id": exp_id, "kind": "mean_std", "seed": "all", "n_seeds": len(group)}
        for key in metric_cols:
            aggregate[key] = float(group[key].mean())
            aggregate[key + "_std"] = float(group[key].std(ddof=1))
        aggregated.append(aggregate)
    save_sheet(pd.concat([table, pd.DataFrame(aggregated)], ignore_index=True), "Final", workbook)
    per_class = pd.DataFrame(class_rows)
    means = per_class.groupby(["exp_id", "class"], sort=False)[["support", "precision", "recall", "f1"]].mean().reset_index()
    stds = per_class.groupby(["exp_id", "class"], sort=False)[["precision", "recall", "f1"]].std().add_suffix("_std").reset_index()
    means = means.merge(stds, on=["exp_id", "class"])
    means["seed"] = "mean_std"
    save_sheet(pd.concat([per_class, means], ignore_index=True), "PerClass", workbook)
    latency_table = pd.DataFrame(latencies)
    if Path("latency.csv").exists():
        previous = pd.read_csv("latency.csv")
        previous = previous.loc[~previous.exp_id.isin(["F01", "T00"])]
        latency_table = pd.concat([previous, latency_table], ignore_index=True)
    save_sheet(latency_table, "Latency", workbook)
    final_stats = next(item for item in aggregated if item["exp_id"] == "F01")
    base_stats = next(item for item in aggregated if item["exp_id"] == "T00")
    delta = final_stats["test_macro_f1"] - base_stats["test_macro_f1"]
    noise = max(final_stats["test_macro_f1_std"], base_stats["test_macro_f1_std"])
    print(f"F01 test macro-F1 = {final_stats['test_macro_f1']:.4f} ± {final_stats['test_macro_f1_std']:.4f}")
    print(f"T00 test macro-F1 = {base_stats['test_macro_f1']:.4f} ± {base_stats['test_macro_f1_std']:.4f}")
    print(f"Δ={delta:+.4f}; std lớn hơn={noise:.4f}. " +
          ("Cải thiện lớn hơn std quan sát." if delta > noise else "Chưa có cải thiện vượt std quan sát."))
    write_json(Path(final_cfg.out_dir) / "final_summary.json", {"groups": aggregated, "delta": delta, "noise_std": noise})
    plot_final_matrices(matrices)
    return pd.concat([table, pd.DataFrame(aggregated)], ignore_index=True)


def plot_final_matrices(matrices):
    import matplotlib.pyplot as plt
    Path("curves").mkdir(exist_ok=True)
    for exp_id in ("F01", "T00"):
        # Gộp số đếm qua 3 seed để nhìn loại nhầm lẫn, không coi là ảnh mới.
        cm = sum(matrix for tag, _, matrix in matrices if tag == exp_id)
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(cm, cmap="Blues")
        for i in range(dataset.NUM_CLASSES):
            for j in range(dataset.NUM_CLASSES):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
        ax.set(xticks=range(9), yticks=range(9), xticklabels=dataset.CLASS_NAMES,
               yticklabels=dataset.CLASS_NAMES, xlabel="Lớp dự đoán", ylabel="Lớp thật",
               title=f"{exp_id}: ma trận nhầm lẫn test cộng qua seed")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(f"curves/{exp_id}_test_confusion.png", dpi=150)
        plt.close(fig)
