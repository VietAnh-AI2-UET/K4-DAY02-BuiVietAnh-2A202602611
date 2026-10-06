# Báo cáo Lab Day 2 — DeepWeeds

## 1. Tóm tắt

So sánh 5 backbone, 10 cấu hình train và 5 cách suy luận trên val.
F01 trên test: macro-F1 0.9727 ± 0.0006; top-1 0.9785 ± 0.0008.
Mốc T00: macro-F1 0.9738 ± 0.0026; Δ=-0.0011; std lớn hơn=0.0026. Chưa có cải thiện vượt std quan sát.
ECE test trước/sau hiệu chỉnh: 0.0084/0.0056. Số seed: 3.

## 2. Dữ liệu và thiết lập

Fold 0, giữ nguyên CSV chia sẵn; chỉ val được dùng để chọn cấu hình.
| class | train | val | test |
| --- | --- | --- | --- |
| Chinee Apple | 675 | 225 | 226 |
| Lantana | 637 | 213 | 213 |
| Parkinsonia | 618 | 206 | 207 |
| Parthenium | 613 | 204 | 205 |
| Prickly Acacia | 637 | 212 | 213 |
| Rubber Vine | 605 | 202 | 202 |
| Siam Weed | 644 | 215 | 215 |
| Snake Weed | 609 | 203 | 204 |
| Negatives | 5463 | 1821 | 1822 |

Macro-F1 coi các lớp quan trọng như nhau; top-1 là tỉ lệ dự đoán đúng.
ECE đo độ lệch giữa sự tự tin của dự đoán và tỉ lệ đúng; giá trị thấp hơn tốt hơn.

Cấu hình chung kết:

```json
{
  "backbone": "convnext_tiny.in12k_ft_in1k",
  "init": "finetune",
  "aug": "randaug",
  "loss": "ce",
  "sampler": null,
  "mix": null,
  "ema_decay": null,
  "img_size": 224,
  "epochs": 12,
  "batch_size": 64,
  "lr_backbone": 0.0001,
  "lr_head": 0.001,
  "weight_decay": 0.05,
  "warmup_epochs": 1.0
}
```

Phần cứng/điều kiện đo:

| gpu | dtype | batch | img_size | torch | preprocessing |
| --- | --- | --- | --- | --- | --- |
| Tesla T4 | fp32 | 1 | 224 | 2.11.0+cu128 | Không tính đọc/resize/normalize; có tính tạo view và gộp xác suất |
| Tesla T4 | fp32 | 32 | 224 | 2.11.0+cu128 | Không tính đọc/resize/normalize; có tính tạo view và gộp xác suất |

## 3. So sánh backbone

| exp_id | backbone | weight_tag | params_M | GMAC | val_macro_f1 | val_top1 | train_seconds_per_epoch | latency_batch1_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B01 | resnet50 | resnet50.a1_in1k | 23.5265 | 4.1095 | 0.8060 | 0.8598 | 47.1720 | 7.7972 |
| B02 | convnext_tiny | convnext_tiny.in12k_ft_in1k | 27.8270 | 4.4697 | 0.9726 | 0.9783 | 55.9182 | 12.6614 |
| B03 | deit_small_patch16_224 | deit_small_patch16_224.fb_in1k | 21.6691 | 4.2503 | 0.9494 | 0.9643 | 38.6272 | 15.0245 |
| B04 | efficientnet_b0 | efficientnet_b0.ra_in1k | 4.0191 | 0.3981 | 0.7522 | 0.8195 | 31.8509 | 9.8247 |
| B05 | mobilenetv3_large_100 | mobilenetv3_large_100.ra_in1k | 4.2136 | 0.2242 | 0.7027 | 0.7843 | 23.2080 | 7.5271 |

B02 đạt macro-F1 val cao nhất 0.9726.
GMAC tính bằng fvcore, không đếm mọi phép toán; độ trễ ở bước 1 mới là số đo sơ bộ.

## 4. Công thức huấn luyện

T00 là công thức nền trên backbone đã chọn; B01 là mốc ResNet của bước 1.
| exp_id | axis | changed | val_macro_f1 | delta_vs_T00 |
| --- | --- | --- | --- | --- |
| T00 | baseline | Công thức nền trên backbone đã chọn | 0.9726 | 0.0000 |
| T01 | B | Thêm đổi màu ảnh | 0.9663 | -0.0063 |
| T02 | B | RandAugment: biến đổi ảnh ngẫu nhiên | 0.9763 | 0.0036 |
| T03 | C | Làm mềm nhãn 0.1 | 0.9712 | -0.0015 |
| T04 | C | Focal: chú ý ảnh khó, gamma=2 | 0.9704 | -0.0023 |
| T05 | C | CE có trọng số nghịch số ảnh train | 0.9650 | -0.0077 |
| T06 | D | Lấy mẫu cân bằng lớp | 0.9716 | -0.0010 |
| T07 | F | EMA: làm mượt trọng số, decay=0.99 | 0.9729 | 0.0003 |
| T08 | B | CutMix: trộn vùng ảnh cùng nhãn | 0.9718 | -0.0008 |
| T09 | combination | T02+T07 | 0.9743 | 0.0016 |

Cấu hình tốt nhất trên val là T02. Kết hợp T09 có Δ=+0.0016 so với T00.
Các thí nghiệm T01–T09 chỉ có một seed, nên chưa xác định được độ dao động riêng của từng thay đổi.
Không dùng std của chung kết để khẳng định một kỹ thuật riêng lẻ thắng chắc chắn.

## 5. Suy luận và tốc độ

| exp_id | description | K | val_macro_f1 | val_ece | p95_ms | relative_cost |
| --- | --- | --- | --- | --- | --- | --- |
| I00 | 1 view - mốc | 1 | 0.9763 | 0.0071 | 12.9752 | 1.0000 |
| I01 | Lật ngang, gộp xác suất | 2 | 0.9760 | 0.0084 | 26.6947 | 2.0998 |
| I02 | 5 crop, gộp xác suất | 5 | 0.9776 | 0.0059 | 62.4658 | 5.1118 |
| I03 | Lật ngang, gộp điểm số trước softmax | 2 | 0.9760 | 0.0097 | 28.2802 | 2.2447 |
| I04 | Hiệu chỉnh xác suất bằng nhiệt độ T | 1 | 0.9763 | 0.0040 | 14.4315 | 1.0866 |

![Chất lượng và độ trễ](curves/inference_tradeoff.png)

Kiểu view đã chốt: `fivecrop`. T được khớp riêng trên val của mỗi seed, rồi giữ cố định trên test.
ECE val sau khớp T được đo trên chính tập khớp; hiệu quả trên dữ liệu mới được kiểm tra bằng ECE test.

## 6. Chung kết và phân tích lỗi

| exp_id | val_macro_f1 | val_macro_f1_std | test_macro_f1 | test_macro_f1_std | test_top1 | test_top1_std | test_ece |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | 0.9734 | 0.0037 | 0.9727 | 0.0006 | 0.9785 | 0.0008 | 0.0056 |
| T00 | 0.9704 | 0.0020 | 0.9738 | 0.0026 | 0.9791 | 0.0017 | 0.0093 |

| exp_id | class | support | precision | recall | f1 | f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | Chinee Apple | 226 | 0.9708 | 0.9263 | 0.9478 | 0.0085 |
| F01 | Lantana | 213 | 0.9843 | 0.9750 | 0.9796 | 0.0048 |
| F01 | Parkinsonia | 207 | 0.9730 | 0.9839 | 0.9784 | 0.0047 |
| F01 | Parthenium | 205 | 0.9983 | 0.9626 | 0.9801 | 0.0051 |
| F01 | Prickly Acacia | 213 | 0.9479 | 0.9687 | 0.9582 | 0.0041 |
| F01 | Rubber Vine | 202 | 0.9851 | 0.9802 | 0.9826 | 0.0050 |
| F01 | Siam Weed | 215 | 0.9803 | 0.9938 | 0.9870 | 0.0080 |
| F01 | Snake Weed | 204 | 0.9558 | 0.9542 | 0.9550 | 0.0107 |
| F01 | Negatives | 1822 | 0.9825 | 0.9885 | 0.9855 | 0.0009 |
| T00 | Chinee Apple | 226 | 0.9687 | 0.9587 | 0.9637 | 0.0047 |
| T00 | Lantana | 213 | 0.9398 | 0.9890 | 0.9636 | 0.0153 |
| T00 | Parkinsonia | 207 | 0.9839 | 0.9807 | 0.9822 | 0.0062 |
| T00 | Parthenium | 205 | 0.9951 | 0.9772 | 0.9860 | 0.0057 |
| T00 | Prickly Acacia | 213 | 0.9351 | 0.9797 | 0.9566 | 0.0134 |
| T00 | Rubber Vine | 202 | 0.9820 | 0.9851 | 0.9836 | 0.0074 |
| T00 | Siam Weed | 215 | 0.9712 | 0.9907 | 0.9808 | 0.0026 |
| T00 | Snake Weed | 204 | 0.9624 | 0.9608 | 0.9616 | 0.0031 |
| T00 | Negatives | 1822 | 0.9911 | 0.9804 | 0.9857 | 0.0019 |

![Ma trận nhầm lẫn F01](curves/F01_test_confusion.png)

| count_across_seeds | true | predicted |
| --- | --- | --- |
| 31 | Chinee Apple | Negatives |
| 23 | Negatives | Prickly Acacia |
| 21 | Snake Weed | Negatives |
| 16 | Chinee Apple | Snake Weed |
| 12 | Rubber Vine | Negatives |

Số nhầm lẫn ở bảng trên cộng qua các seed, không phải số ảnh khác nhau.
Minh hoạ 9/9 ảnh đoán sai của seed 0; không dùng các ảnh này để chọn lại cấu hình.

![Ảnh đoán sai](curves/F01_error_examples.png)

| Filename | true | predicted | confidence |
| --- | --- | --- | --- |
| 20170630-155129-0.jpg | Snake Weed | Negatives | 0.9876 |
| 20170630-155109-0.jpg | Snake Weed | Negatives | 0.9801 |
| 20170706-113305-0.jpg | Snake Weed | Negatives | 0.9506 |
| 20170711-113440-0.jpg | Snake Weed | Negatives | 0.9497 |
| 20170705-162907-0.jpg | Chinee Apple | Snake Weed | 0.9211 |
| 20170207-153455-0.jpg | Chinee Apple | Snake Weed | 0.9108 |
| 20170705-162506-0.jpg | Snake Weed | Chinee Apple | 0.8981 |
| 20170627-105314-0.jpg | Chinee Apple | Negatives | 0.8559 |
| 20170718-133537-2.jpg | Chinee Apple | Negatives | 0.8546 |
Các cặp nhầm phổ biến gợi ý cần xem kỹ vùng cây và nền ảnh; chưa đủ bằng chứng để kết luận nguyên nhân sinh học.

## 7. Kết luận

Δ F01 so với T00 = -0.0011. Chưa có cải thiện vượt std quan sát.
F01 có p95 lớn nhất qua seed = 66.41 ms ở batch 1. Đạt ngưỡng p95 ≤ 100 ms trong điều kiện đã đo.
ECE test giảm sau hiệu chỉnh (0.0084 → 0.0056).
Chênh lệch qua backbone, recipe và cách suy luận chỉ là quan sát ở các điều kiện khác nhau; không cộng chúng như các đóng góp độc lập.

## 8. Hạn chế

Một fold; các vòng chọn cấu hình chỉ có một seed. Trọng số pretrained có thể có công thức huấn luyện khác nhau.
Độ trễ không tính đọc/resize/normalize ảnh. Điều kiện phần cứng và tải máy có thể ảnh hưởng kết quả.
Không thay đổi cấu hình sau khi xem test. Việc tiếp theo cần dùng dữ liệu mới hoặc một quy trình đánh giá độc lập.

## 9. Phụ lục

Top 10 cấu hình xếp theo val, không xếp theo test:

| rank | exp_id | source | backbone | n_seeds | val_macro_f1 | val_macro_f1_std | val_top1 | params_M | GMAC_estimate | train_seconds_per_epoch | latency_batch1_ms | p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | I02 | Inference | convnext_tiny.in12k_ft_in1k | 1 | 0.9776 | — | 0.9831 | 27.8270 | 22.3484 | — | — | 62.4658 |
| 2 | I00 | Inference | convnext_tiny.in12k_ft_in1k | 1 | 0.9763 | — | 0.9823 | 27.8270 | 4.4697 | — | — | 12.9752 |
| 3 | I04 | Inference | convnext_tiny.in12k_ft_in1k | 1 | 0.9763 | — | 0.9823 | 27.8270 | 4.4697 | — | — | 14.4315 |
| 4 | T02 | Training | convnext_tiny.in12k_ft_in1k | 1 | 0.9763 | — | 0.9823 | 27.8270 | 4.4697 | 55.3530 | 14.0254 | — |
| 5 | I01 | Inference | convnext_tiny.in12k_ft_in1k | 1 | 0.9760 | — | 0.9820 | 27.8270 | 8.9394 | — | — | 26.6947 |
| 6 | I03 | Inference | convnext_tiny.in12k_ft_in1k | 1 | 0.9760 | — | 0.9820 | 27.8270 | 8.9394 | — | — | 28.2802 |
| 7 | T09 | Training | convnext_tiny.in12k_ft_in1k | 1 | 0.9743 | — | 0.9809 | 27.8270 | 4.4697 | 56.3279 | 12.5785 | — |
| 8 | F01 | Final | convnext_tiny.in12k_ft_in1k | 3 | 0.9734 | 0.0037 | 0.9797 | 27.8270 | 22.3484 | 55.4131 | — | 64.3476 |
| 9 | T07 | Training | convnext_tiny.in12k_ft_in1k | 1 | 0.9729 | — | 0.9783 | 27.8270 | 4.4697 | 55.6511 | 13.5575 | — |
| 10 | B02 | Backbones | convnext_tiny | 1 | 0.9726 | — | 0.9783 | 27.8270 | 4.4697 | 55.9182 | 12.6614 | — |

GMAC_estimate trong Summary là GMAC/forward nhân số view, chưa tính chi phí gộp; số đo sơ bộ latency_batch1_ms tách riêng p95_ms đo đủ 50 lần.

Cấu hình/log/summary được chép vào `evidence/runs/`; kết quả tính lại bằng eval.py ở `eval_out/`.
Các bản sao CSV, biểu đồ và code có mã SHA-256 trong `artifact_manifest.json` để kiểm tra file thay đổi.
README hướng dẫn chạy lại; notebook ở `code/lab_day2.ipynb`.