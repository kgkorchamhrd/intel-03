# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: ???
./splitted_dataset/train: ???​
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/val: ???
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/train/<class#>: ???​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2|0.996|3.64|13.53|64|0.0008|-|
|EfficientNet-B0|0.593|5.81|4.03|64|0.002|-| 
|DeiT-Tiny|0.192|7.46|2.44|64|0.000049|-| 
|MobileNet-V3-large-1x|0.992|7.63|11.01|64|0.0003|-| 


## FPS 측정 방법
