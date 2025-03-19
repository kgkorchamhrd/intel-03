# Homework03
Smart Factory 불량 분류 모델 Training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: ???
./splitted_dataset/train: ???
./splitted_dataset/train/<class#>: ???
./splitted_dataset/train/<class#>: ???
./splitted_dataset/val: ???
./splitted_dataset/val/<class#>: ???
./splitted_dataset/val/<class#>: ???
```

## Training 결과
| Classification model        | Accuracy | FPS | Training time | Batch size | Learning rate | Other hyper-params |
|----------------------------|----------|-----|--------------|------------|--------------|--------------------|
| MobileNet-V3-small         | 17.41%   |1.93 | 35m 26s      | 10          | 0.01         | Momentum: 0.9      |
| MobileNet-V3-large         | 24.50%   |0.11 | 2h 5m 46s    | 10          | 0.0058       | Momentum: 0.9      |



## FPS 측정 방법




