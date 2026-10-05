"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.

    TODO:
      - đọc ba file CSV bằng pandas
      - trả về (train_df, val_df, test_df)
    """
    labels_dir = Path(labels_dir)
    return tuple(pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
                 for split in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    TODO kiểm tra, mỗi ý lỗi thì `assert` / raise để dừng ngay:
      1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. hợp ba tập phải bằng đúng 17.509 ảnh
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict, ví dụ {"n": {...}, "per_class": {...}, "overlap": {...}} để dán vào báo cáo.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    filenames, per_class = {}, {}
    for name, df in splits.items():
        required = {"Filename", "Label"}
        if not required.issubset(df.columns):
            raise ValueError(f"{name}: thiếu cột {required - set(df.columns)}")
        if df[list(required)].isna().any().any():
            raise ValueError(f"{name}: có giá trị trống")
        if not df["Label"].isin(range(NUM_CLASSES)).all():
            raise ValueError(f"{name}: Label phải là số nguyên từ 0 đến 8")
        if df["Filename"].duplicated().any():
            raise ValueError(f"{name}: có Filename trùng")
        filenames[name] = set(df["Filename"])
        per_class[name] = df["Label"].value_counts().reindex(
            range(NUM_CLASSES), fill_value=0).astype(int).to_dict()
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = filenames[left] & filenames[right]
        overlap[f"{left}_{right}"] = len(common)
        if common:
            raise ValueError(f"{left} và {right}: trùng {len(common)} ảnh")
    all_filenames = set().union(*filenames.values())
    if len(all_filenames) != 17509:
        raise ValueError(f"Tổng phải là 17.509 ảnh, nhận được {len(all_filenames):,}")
    images_dir = Path(images_dir)
    missing = sorted(name for name in all_filenames if not (images_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh trong {images_dir}: {missing[:5]}")
    counts = {name: len(df) for name, df in splits.items()}
    stats = {"n": counts, "per_class": per_class, "overlap": overlap,
             "total": len(all_filenames)}
    for name, count in counts.items():
        print(f"{name}: {count:,} ảnh ({count / len(all_filenames):.2%})")
    print(pd.DataFrame(per_class).rename(index=dict(enumerate(CLASS_NAMES))))
    print(f"Tổng: {len(all_filenames):,} ảnh; giao giữa các tập: {overlap}")
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị.

    Gợi ý các giá trị `aug` (trục B của GUIDE.md mục 3): "basic", "color", "trivial", "randaug".
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: ảnh gốc 256x256 -> CenterCrop(img_size) (hoặc giữ nguyên 256; ghi rõ bạn chọn gì)
              + ToTensor + Normalize. KHÔNG augmentation ngẫu nhiên khi đánh giá.

    TODO: dùng torchvision.transforms (hoặc v2). Lưu ý: lật dọc có hợp lệ với ảnh cỏ dại không?
    """
    operations = []
    if train:
        if aug not in {"basic", "color", "trivial", "randaug"}:
            raise ValueError(f"Không hỗ trợ augmentation: {aug}")
        # Crop và lật chỉ thay đổi ảnh; nhãn loài cây vẫn giữ nguyên.
        operations += [transforms.RandomResizedCrop(img_size),
                       transforms.RandomHorizontalFlip()]
        if aug == "color":
            operations.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
        elif aug == "trivial":
            operations.append(transforms.TrivialAugmentWide())
        elif aug == "randaug":
            operations.append(transforms.RandAugment())
    else:
        operations += [transforms.Resize(256), transforms.CenterCrop(img_size)]
    # Đổi ảnh thành tensor [C,H,W], rồi chuẩn hoá theo trọng số ImageNet.
    operations += [transforms.ToTensor(),
                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(operations)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) phải trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.

    TODO:
      - __init__(self, df, images_dir, transform): giữ df, mở ảnh bằng PIL, chuyển sang RGB
      - __len__
      - __getitem__ -> (tensor, int(label), filename)
      - (tuỳ chọn) nạp trước ảnh vào RAM nếu bị nghẽn đọc đĩa trên Colab
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        # Lấy tên ảnh và nhãn từ CÙNG một dòng, tránh ghép nhầm khi shuffle.
        row = self.df.iloc[i]
        filename, label = str(row["Filename"]), int(row["Label"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, filename


def seed_worker(worker_id: int) -> None:
    # PyTorch cấp seed khác nhau cho từng worker; dùng lại cho NumPy và random.
    # Đặt hàm ở cấp module để DataLoader dùng được trên Windows.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader.

    TODO:
      - train=True: shuffle (hoặc dùng sampler); train=False: không shuffle, giữ thứ tự df
        (thứ tự phải ổn định để ghép logit với Filename)
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler với trọng số
        1/(số ảnh của lớp) (trục D của GUIDE.md mục 3)
      - drop_last=True khi train nếu batch cuối quá nhỏ làm BatchNorm không ổn định
      - pin_memory=True, num_workers hợp lý; seed cho worker (worker_init_fn) để tái lập
    """
    if sampler not in {None, "balanced"}:
        raise ValueError(f"Không hỗ trợ sampler: {sampler}")
    if sampler is not None and not train:
        raise ValueError("Sampler cân bằng chỉ dùng cho tập train")
    ds = DeepWeedsDataset(df, images_dir, transform)
    # Generator riêng giúp thứ tự batch và seed worker tái lập sau set_seed().
    generator = torch.Generator().manual_seed(torch.initial_seed())
    sample_strategy = None
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        weights = df["Label"].map(1.0 / counts).to_numpy(copy=True)
        sample_strategy = WeightedRandomSampler(
            weights, len(df), replacement=True, generator=generator)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train and sample_strategy is None,
        sampler=sample_strategy, num_workers=num_workers,
        drop_last=train, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=generator,
    )
