---
title: "VGGNet Review"
author: "유원규"
---

# VGGNet  

VGGNet은 19 layer를 지닌 깊은 network로 ILSVRC 2014 대회에서 2등을 차지했습니다. 역대 ILSVRC 우승작 network의 깊이는 8 layer에 불가했습니다. 깊이가 깊어짐에 따라 overfitting, gradient vanishing, 연산량 문제가 생기기 때문에 깊이를 증가시키는 것이 쉬운 문제는 아니었습니다.  

![1번사진](https://user-images.githubusercontent.com/83005178/170999175-1073fc3e-8464-47d6-b64a-0e840d194f05.png)  

<br>


VGGNet 논문에 나와있는 핵심 내용을 간추려 보았습니다.

1. 깊이를 증가하면 정확도가 좋아집니다.  
2. 3x3 filter를 여러 겹 사용하여 5x5, 7x7 filter를 분해하면 추가적인 비선형성을 부여하고 parameter의 수를 감소시킵니다.  
3. pre-initialization을 이용하면 모델이 빠르게 수렴합니다.  
4. data augmentation(resize, crop, flip)을 적용하면 다양한 scale로 feature를 포착할 수 있습니다.  
5. 빠른 학습을 위해 4-GPU data parallerism을 활용했습니다.  

---

## Abstarct

VGG 논문은 convolutional networt의 깊이를 증가시키면서 정확도 양상을 조사합니다. 
모델은 3x3 convolution layer를 연속적으로 사용한 16-19 깊이의 모델입니다.  

실제로 이 모델로 ImageNet Challenge 2014 의 classification 분야에서 2등을 차지했습니다. 깊이가 깊어짐에 따라 overfitting과 gradient vanishing 문제가 생기는데, 어떻게 문제점을 해결하고 대회 2등을 차지할 수 있는지 살펴보도록 하겠습니다.  

### Introduction

 Convolutional networks (ConvNets)는 이미지, 비디오 인식 분야에서 큰 성공을 거두었고 이것은 ImageNet과 같은 큰 공공의 이미지 저장소와 고성능 컴퓨팅 시스템(GPU) 덕분이라고 말합니다. 특히 시각 인식 구조의 발전은 large-scale 이미지 분류 시스템(AlexNet, ZFNet 등)에 대한 시험대 역활을 하는 ILSVRC 대회가 큰 역할을 하고 있다고 합니다.

 ConvNets이 computer vision 분야에서 더 상용화되가면서, 더 좋은 정확도를 얻기 위해 AlexNet의 기본 구조를 향상시키기 위한 많은 시도가 이루어 졌습니다. 이 논문에서는 ConvNet 구조의 깊이에 집중합니다. 이를 위해 더 많은 convolutional layer를 추가하였고 이것은 모든 layers에 매우 작은 3x3 convolution filters의 사용 덕분에 가능하다고 말합니다.
 결과적으로, 상당히 더 정확한 ConvNet 구조를 구상했으며 높은 정확도를 갖고 다른 이미지 인식 dataset에도 적용가능하다고 합니다.  

### ConvNet Configurations

 공정한 설정으로 ConvNet의 깊이가 증가함에 따른 성능 향상을 측정하기 위해 모든 conv layer의 파라미터를 동일하게 설정합니다.

#### Architecture

ConvNet의 입력값은 고정된 크기의 224x224 RGB 이미지이며 전처리는 training set의 각 pixel에 평균 RGB 값을 빼주었습니다.

입력 이미지는 3x3 filter가 적용된 ConvNet에 전달되고, 또한 비선형성을 위해 1x1 convolutional filters도 적용합니다. stride=1이 적용되었고 공간 해상도를 보존하기 위해 padding을 적용합니다. 일부 conv layer에는 max-pooling(size=2x2, stride=2) layer를 적용합니다.

convolutional layers 다음에는 3개의 Fully-Connected(FC) layer가 있고, 첫 번째와 두 번째 FC는 4096 channel, 세 번째 FC는 1000 channels를 갖고 있는 soft-max layer 입니다. 모든 hidden layer에는 활성화 함수로 ReLU를 이용했으며 AlexNet에 적용된 LRN(Local Response Normalization)는 VGG 모델의 성능에 영향이 없기 때문에 적용하지 않았습니다.  

#### Configurations  

Table 1에 깊이를 서로 달리한 모델 A~E의 배치가 나와있습니다. 11~19 범위의 깊이로 실험이 진행되었고, 넓이(channels)는 각 max-pooling layer 이후에 2배씩 증가하여 512에 도달합니다. Table2는 각 배치의 parameter 수가 나와있습니다.  

![2번사진](https://user-images.githubusercontent.com/83005178/170999617-f2d58ef4-6650-4891-951b-5a69518e85c8.png)  

<br>

#### Discussion

VGG model은 ILSVRC-2012 우승작 AlexNet, ILSVRC-2013 우승작 ZFNet과 매우 다릅니다. 둘 다 큰 필터 사이즈를 사용했지만, VGG는 전체에 stride=1인 3x3 필터 사이즈만을 사용합니다. 이것으로 엄청난 사실을 발견합니다.  

3x3 convolutional filter를 2개 이용하면 5x5 convolutional, 3개 이용하면 7x7 convolutional가 됩니다. 3x3 filter를 여러 겹 이용하게 되면 하나의 relu 대신 2개, 3개의 relu를 이용할 수 있고, parameter 수를 감소시킬 수 있습니다. 예를 들어, C개의 channel를 가진 3개의 3x3 filter를 이용하면 연산량은 3(3 2 C 2 3 2 C 2 ) = 27 C 2
가 되고, C 개의 channel를 가진 1개의 7x7 filter를 이용하면 연산량은 7 2 C 2 7 2 C 2 =49 C 2 가 됩니다.  

![3번사진](https://user-images.githubusercontent.com/83005178/170999802-8f0aef85-1c88-4ba9-a08c-b66f6bc4cb6d.png)  

7x7 filter를 3개의 3x3 filter로 분해하면 parameter 수도 감소시키고 더 많은 relu 함수를 이용할 수 있게 됩니다.

1x1 conv layer는 비선형성을 부여하기 위한 용도입니다. 입력과 출력의 channels를 동일하게 하고 1x1 conv layer를 이용하면 relu 함수를 거치게 되어 추가적인 비선형성이 부여됩니다.  

### Classification Framwork  

이번 section에서는 training과 evaluation의 세부사항에 대해 설명합니다.

#### Training
학습 과정은 입력 이미지 crop 방법을 제외하고 AlexNet과 동일하게 진행되었습니다. 즉, momentum이 있는 mini-batch gradient descent를 사용하여 다항 회기 분석을 최적화함으로써 학습이 진행되었습니다. 학습 hyper parameter는 다음과 같습니다. 
batch size = 256, momentum = 0.9, weight decay = 0.00005, epoch = 74, learning rate = 0.01(10배씩 감소)

논문에서는 AlexNet과 비교하여 우리의 모델이 깊고 더 많은 parameter를 갖고 있음에도 3x3 filter size의 여러겹 이용한 것과 특정 layers에서 pre-initialisation 덕분에 수렴하기 위한 적은 epoch를 요구한다고 합니다. 논문에서 말하는 pre-initialisation이 뭔지 알아보도록 하겠습니다.

- **pre-initialisation**
bad initialisation은 gradine의 불안정함 때문에 학습을 지연시킵니다. 이를 해결하기 위해 pre-initialisation을 이용했습니다. 가장 얇은 구조인 A를 학습시킨 이후에 학습된 첫 번째, 네 번째 Conv layer와 3개의 FC layer의 가중치를 이용하여 다른 깊은 모델을 학습시켰다고 합니다. 미리 가중치가 설정되어 수렴하기 까지의 적은 epoch가 필요했었던 것이었습니다. 그리고 가중치 초기화 값은 평균 0 분산 0.01인 정규 분포에서 무작위로 추출했다고 합니다.  

<br>

- **data augmentation**

VGG 모델은 3가지의 data augmentation이 적용되었습니다.

(1) crop된 이미지를 무작위로 수평 뒤집기
(2) 무작위로 RGB 값 변경하기
(3) image rescaling

실험을 위해 3가지 방법으로 rescale을 하고 비교를 합니다.
  - input size = 256, 256로 고정  
  
  - input size = 356 356로 고정  

  - 입력 size를 [256, 512] 범위로 랜덤하게 resize 합니다. 이미지 안의 object가 다양한 규모로 나타나고, 다양한 크기의 object를 학습하므로 training에 효과가 있었다고 합니다.  
  빠른 학습을 위해서 동일한 배치를 갖은 size=384 입력 image로 pre-trained 된 것을 fine-tunning함으로써 multi-scale 모델을 학습시켰습니다.  


#### Testing
 test image를 다양하게 rescale 하여 trained ConvNet에 입력으로 이용합니다. 다양하게 rescale 함으로써 성능이 개선되었다고 합니다. 또, 수평 뒤집기를 적용하여 test set을 증가시켜주었고 최종 점수를 얻기 위해 동일한 이미지에 data augmentation이 적용된 이미지들의 평균 점수를 이용했다고 합니다.
 test image에 crop도 적용했습니다. 이는 더 나은 성능을 보였지만 연산량 관점에서 비효율적이라고 말합니다.

#### Implementation Details

2013년 12월에 출시된 C++ Caffe 를 이용해서 구현되었다고 합니다.
 4-GPU system에서 학습 시간은 2~3주가 소요되었고 이는 a single GPU보다 3.75배 빠르다고 합니다.
 4-GPU system은 data parallelism을 이용하는 Multi-GPU training입니다. training 이미지의 각 배치가 다수의 GPU에서 수행되고 각각의 GPU에서 병렬하게 처리됩니다. GPU 배치 기울기가 계산된 후 모든 배치에 대한 기울기를 얻기 위해 평균을 이용합니다.  

### Classification experiments

- Dataset

dataset은 1000개의 클래스에 대한 이미지들이 포함되어 있고, 이것은 3가지 set(training:1.3M, validation:50K, testing:1000K)으로 나누었습니다.  
평가기준은 top-1, top-5 error를 이용하였으며 전자는 예측이 잘못된 이미지의 비율, 후자는 top-5 예측된 범주에 정답이 없는 이미지의 비율입니다.  
실험에서는 validation set을 test set으로 이용했습니다.

#### Single Scale evaluation  

test set의 size가 고정된 단일 규모 평가입니다.  
![4번사진](https://user-images.githubusercontent.com/83005178/171000481-aadd944e-16ff-41e7-9e5e-b5144264d2b8.png)  

1. AlexNet에서 이용되었던 LRN(local response normalisation)이 효과가 없었습니다.  

2. 깊이가 깊어지 수록 error가 감소한다는 것을 관찰했습니다. C가 D보다 성능이 낮고 B보다 성능이 높습니다.  이는 추가적인 비선형성은 도움이 되는 반면에 conv filter를 사용하여 공간적인 맥락을 포착하는 것이 중요하다는 것을 나타냅니다.  
그리고 5x5 filter를 이용하는 것보다 3x3 filter를 2개 이용하는 것이 성능 향상에 도움이 된다고 합니다.  

3. 다양한 scale[256~512]로 resize한 것이 고정된 scale의 training image보다 성능이 좋았습니다.  

#### Multi-scale evaluation  

test set의 다양한 scale에 대한 실험입니다.  

실험 결과 test 이미지를 다양한 scale로 resize했을 때, 단일 size보다 더 나은 성능을 보였습니다.  
![5번사진](https://user-images.githubusercontent.com/83005178/171000788-c9c6da2a-6b5c-48fe-9b32-5e0d0f025f2b.png)  

#### Multi-crop evalutation  

아래 표에 보이는 것처럼 test 이미지를 다양하게 crop을 해주어 더 나은 성능을 얻었다고 합니다.  
![6번사진](https://user-images.githubusercontent.com/83005178/171000880-661e0335-1e83-448c-a1ba-365b01156f6f.png)  


#### ConvNet fusion  

앙상블에 대한 내용입니다. 모델 7개를 앙상블한 ILSVRC 제출물은 test set top-5 error가 7.5% 나왔고, 추후에 모델 2개를 앙상블하여 test set top-5 error를 6.8% 까지 낮추었다고 합니다.  
![7번사진](https://user-images.githubusercontent.com/83005178/171000979-42b54277-9f82-41de-98b0-63c0e2815af5.png)

---

# **References**

['Very Deep Convolutional Networks for large-scale image recognition'(VGGNet)](https://arxiv.org/pdf/1409.1556.pdf)

[VGGNet](https://phil-baek.tistory.com/entry/1-Very-Deep-Convolutional-Networks-for-Large-Scale-Image-Recognition-VGGNet-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)