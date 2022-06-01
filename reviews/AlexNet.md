---
title: "AlexNet Review"
author: "유원규"
---

# AlexNet
컴퓨터 비전 분야의 ‘올림픽’ 이라 할 수 있는 ILSVRC(ImageNet Large-Scale Vosual Recognition Challenge)의 2012년 대회에서 AlexNet이 Top 5 test error 기준 15.4%를 기록해 2위(26.6%)를 큰 폭으로 따돌리고 1위를 차지했습니다. (Top 5 test error란 모델이 예측한 최상위 5개 범주 가운데 정답이 없는 경우의 오류율을 말합니다.)이 AlexNet덕분에 딥러닝, 특히 CNN이 세간의 주목을 받게 되었으며, CNN 구조의 GPU 구현과 dropout 적용이 보편화 되었습니다.  

---

## The Dataset  


![1번사진](https://user-images.githubusercontent.com/83005178/170994012-54a32d84-2ef6-4d5f-85a6-fcf4d61d0297.png)  



- ImageNet dataset
22,000개 범주로 구성되어 있고 1500만개의 고해상도 이미지가 포함되어있는 data set입니다.
ILSVRC 대회는 ImageNet dataset의 subset을 이용하는데, 각 범주당 1000개의 이미지가 포함되어 있는 1000개 범주를 이용합니다. 따라서, 대략 120만개의 training 이미지와 50,000개의 validation 이미지, 150,000개의 testing 이미지로 구성되었습니다.

- 이미지 크기 256x256으로 고정
이미지를 동일한 크기(256x256)으로 고정시켜줬습니다. 나중에 FC layer의 입력 크기가 고정되어 있어야하기 때문입니다. 만약 입력 이미지의 크기가 다르다면 FC layer에 입력되는 feature 개수가 모두 다르게 됩니다.
resize 방법은 이미지의 넓이와 높이 중 더 짧은 쪽을 256으로 고정시키고 중앙 부분을 256x256 크기로 crop 해주었습니다.
각 이미지의 pixel에 traing set의 평균을 빼서 normalize 해주었습니다.  

---  

## AlexNet Architecture  

AlexNet은 일부가 max-pooling layer가 적용된 5개의 convolutional layer와 3개의 fully-connected layer로 이루어져 있습니다. 아래 그림을 확인하면서 구조에 대해 세부적으로 알아보겠습니다.  

![2번사진](https://user-images.githubusercontent.com/83005178/170994200-96bf7361-9798-49a7-a949-4aa3ae49f94d.png)  


AlexNet은 [Input layer – Conv1 – MaxPool1 – Norm1 – Conv2 – MaxPool2 – Norm2 – Conv3 – Conv4 – Conv5 – Maxpool3 – FC1 – FC2 – Output layer]으로 구성되어 있습니다.  

- [Input layer]
224x224x3 크기의 이미지입니다.
<br>

- [Conv1]
96 kernels of size 11x11, stride=4, padding=0
input : 224x224x3
output : 55x55x96
<br>

- [MaxPool1]
3x3 kernels, stride=2
input : 55x55x96
output : 27x27x96
<br>

- [Norm1]
LRN을 사용한 normalization layer 입니다. LRN이 어떤 것인지는 아래에 설명하겠습니다.
normalization을 수행하는 layer 입니다.
input : 27x27x96
output : 27x27x96
<br>

- [Conv2]
256 kernels of size 5x5, stride=1, Padding=2
논문의 그림에는 3x3 size의 kernel을 사용했다고 나오는데 논문의 그림이 잘못되었다고 합니다
input : 27x27x96
output : 27x27x256
<br>
 
- [MaxPool2]
3x3 kernels, stride = 2
input : 27x27x256
output : 13x13x256
<br>

- [Norm2]
LRN을 사용한 normalization layer
input : 13x13x256
output : 13x13x256
<br>
 
- [Conv3]
384 kernels of size 3x3, stride=1, padding=1
input : 13x13x256
output : 13x13x384
<br>
 
- [Conv4]
384 kernels of size 3x3, stride=1, padding=1
input : 13x13x384
output : 13x13x256
<br>

- [Conv5]
256 kernels of size 3x3, stride=1, padding=1
input : 13x13x384
output : 13x13x256
<br>

- [MaxPool3]
3x3 kernels, stride=2
input : 13x13x256
output : 6x6x256
<br>

- [FC1]
fully connected layer with 4096 neurons
input : 6x6x256
output : 4096
<br>

- [FC2] 
fully connected layer with 4096 neurons
input : 4096
output : 4096
<br>

- [output layer]
fully connected layer with 1000-way softmax
input : 4096
output : 1000
<br>

---

## AlexNet의 구조에 적용된 특징  

- **ReLU Nonlinearity**
활성화 함수로 ReLU를 적용했습니다.
논문에서는 saturating nonlinearity(tanh, sigmoid)보다 non-saturatung nonlinearity(ReLU)의 학습 속도가 몇배는 빠르다고 나와있습니다.

저자는 tanh와 Relu의 학습 속도를 비교하기 위해 실험결과를 논문에 담았습니다. CNN으로 CIFAR-10 dataset을 학습시켰을 때 25% training error에 도달하는 ReLU와 tanh의 epoch수의 실험 결과 그림입니다.  


![3번사진](https://user-images.githubusercontent.com/83005178/170994999-14023753-225d-43ca-844f-5b66b9eab3bd.png)  

4층의 CNN으로 CIFAR-10을 학습시켰을 때 ReLU가 tanh보다 6배 빠르다는 내용입니다.  

- **Training on Multiple GPUs**
network를 2개의 GPU로 나누어서 학습시켰습니다. 이를 GPU parallelization이라고 합니다.
 저자는 120만개의 data를 학습시키기 위한 network는 하나의 GPU로 부족하다고 설명합니다.  
 2개의 GPU로 나누어서 학습시키니 top-1 erroe와 top-5 error가 1.7%, 1.2% 감소되었으며 학습속도도 빨라졌다고 말합니다.  
 예를 들어, 90개의 kernel이 있다고 하면 45개를 GPU 1에 할당하고 남은 45개를 GPU 2에 할당하여 학습합니다.

여기서 추가적인 기법이 있는데, 데이터를 두 개의 GPU로 나누어 학습시키다가 하나의 layer에서만 GPU를 통합시키는 것입니다.  
논문에서는 3번째 Conv layer에서만 GPU를 통합시킨다고 말합니다. 이를 통해 계산량의 허용가능한 부분까지 통신량을 정확하게 조정할 수 있다고 나와있습니다.  

![4번사진](https://user-images.githubusercontent.com/83005178/170995201-476d19d3-9ac6-440b-9ae7-41dabd967745.png)  

위 그림은 GPU 1, GPU 2 각각에서 학습된 kernel map 입니다. GPU 1에서는 색상과 관련 없는 정보를 학습하고 GPU 2는 색상과 관련된 정보를 학습하는 것을 확인할 수 있습니다. 이처럼 각각의 GPU는 독립적으로 학습된다고 나와 있습니다.  

- **Local Response Normalization(LRM)**  
LRN은 generalizaion을 목적으로 합니다.  
sigmoid나 tanh 함수는 입력 date의 속성이 서로 편차가 심하면 saturating되는 현상이 심해져 vanishing gradient를 유발할 수 있게 됩니다.  
반면에 ReLU는 non-saturating nonlinearity 함수이기 때문에 saturating을 예방하기 위한 입력 normalizaion이 필요로 하지 않는 성질을 갖고 있습니다.  
ReLU는 양수값을 받으면 그 값을 그대로 neuron에 전달하기 때문에 너무 큰 값이 전달되어 주변의 낮은 값이 neuron에 전달되는 것을 막을 수 있습니다. 이것을 예방하기 위한 normalization이 LRN입니다.  

논문에서는 LRN을 측면 억제(later inhibition)의 형태로 구현된다고 나와 있습니다.  
측면 억제는 강한 자극이 주변의 약한 자극을 전달하는 것을 막는 효과를 말합니다.  

![5번사진](https://user-images.githubusercontent.com/83005178/170995545-9c9a3c43-9005-4ec7-a216-598fb3f469d6.png)  

위 그림은 측면 억제의 유명한 그림인 헤르만 격자입니다.  
검은 사각형안에 흰색의 선이 지나가고 있습니다. 신기한 것은 흰색의 선에 집중하지 않을 때 회식의 점이 보이는데 이러한 현상이 측면 억제에 의해 발생하는 것입니다.  
이는 흰색으로 둘러싸인 측면에서 억제를 발생시키기 때문에 흰색이 더 반감되어 보입니다.  

AlexNet에서 LRN을 구현한 수식을 살펴보겠습니다.  

![6번사진](https://user-images.githubusercontent.com/83005178/170995722-9ad918b2-5630-4309-8266-06fc9ae38d8f.png)  


a는 x,y 위치에 적용된 i번째 kernel의 output을 의미하고 이 a를 normalization하여 큰 값이 주변의 약한 값에 영향을 주는 것을 최소화 했다고 나와 있습니다.  
이러한 기법으로 top-1와 top-5 eroor를 각각 1.4%, 1.2% 감소시켰다고 합니다.  

하지만 AlexNet 이후 현대의 CNN에서는 local response normalization 대신 batch normalization 기법이 쓰인다고 합니다!  

- **Overlapping Pooling**  
Overlapping pooling을 통해서 overfit을 방지하고 top-1와 top-5 error를 각각 0.4%, 0.3% 낮추었다고 말합니다.  
Pooling layer은 동일한 kernel map에 있는 인접한 neuron의 output을 요약해줍니다.  
전통적으로 pooling layer는 overlap하지 않지만 AlexNet은 overlap을 해주었습니다.  
kernel size는 3, stride는 2를 이용해서 overlap을 해주었다고 나와있습니다.

---

## Reducing Overfitting  

AlexNet에는 6천만개의 parameters가 존재합니다. 이미지를 ILSVRC의 1000개 classes로 분류하기 위해서는 상당한 overfitting 없이 수 많은 parameters를 학습 시키는 것은 어렵다고 말합니다. 논문에서는 overfitting을 피하기 위해 적용한 두 가지 기법을 소개합니다.

- **Data Augmentation**
Data Augmentation은 현재 갖고 있는 데이터를 좀 더 다양하게 만들어 CNN 모델을 학습시키기 위해 만들어진 개념입니다. 이러한 기법은 적은 노력으로 다양한 데이터를 형성하게하여 overfitting을 피하게 만들어 줍니다. 또한 data augmentation의 연산량은 매우 적고 CPU에서 이루어지기 때문에 계산적으로 부담이 없다고 말합니다.

논문에서 2가지 data augmentation 를 적용했다고 나와있습니다.  

- generating image translation and horizontal reflections
이미지를 생성시키고 수평 반전을 해주었다고 합니다. 어떻게 적용하였는지 알아보겠습니다.

 256x256 이미지에서 224x224 크기로 crop을 합니다. crop 위치는 중앙, 좌측 상단, 좌측 하단, 우측 상단, 우측 하단 이렇게 5개의 위치에서 crop을 합니다. crop으로 생성된 5개의 이미지를 horizontal reflection을 합니다. 따라서 하나의 이미지에서 10개의 이미지가 생성됩니다.  

 ![7번사진](https://user-images.githubusercontent.com/83005178/170996117-d22d8101-948c-4e84-aec3-195d9e57e878.png)  
 
 - altering the intensities of the RGB channels in training images
 image의 RGB pixel 값에 변화를 주었습니다. 어떻게 변화를 주었고 어떤 효과가 있었는지 알아보겠습니다.
ImageNet의 training set에 RGB pixel 값에 대한 PCA를 적용했습니다. PCA를 수행하여 RGB 각 생상에 대한 eigenvalue를 찾습니다. eigenvalue와 평균 0, 분산 0.1인 가우시안 분포에서 추출한 랜덤 변수를 곱해서 RGB 값에 더해줍니다.  

![8번사진](https://user-images.githubusercontent.com/83005178/170996221-33382ffa-ed8b-4bac-a0c5-0cb11e62efe4.png)  

이를 통해 조명의 영향과 색의 intensity 변화에 대한 불변성을 지닌다고 합니다. 이 기법으로 top-1 error를 1% 낮추었다고 합니다.  

- **Dropout**  
서로 다른 모델의 예측을 결합하는 앙상블 기법은 test error를 감소시키기 효과적인 방법입니다. 하지만 AlexNet은 학습시키는데에 몇일이 걸려 이것을 적용하기가 힘들었다고 합니다. 따라서 모델 결합의 효과적인 버전인 dropout을 적용시켰다고 합니다.  
dropout의 확률을 0.5로 설정하고 dropout된 neuron은 순전파와 역전파에 영향을 주지 않습니다. 매 입력마다 dropout을 적용시키면 가중치는 공유되지만 신경망은 서로 다른 구조를 띄게 됩니다. neuron은 특정 다른 neuron의 존재에 의존하지 않기 때문에 이 기법은 복잡한 neuron의 co-adaptation를 감소시킨다고 말합니다. 그러므로 서로 다른 neuron의 임의의 부분 집합끼리 결합에 유용한 robust 특징을 배울 수 있다고 말합니다.  
train에서 dropout을 적용시켰고 test에는 모든 neuron을 사용했지만 neuron의 결과값에 0.5 곱해주었다고 합니다.  
또한 AlexNet은 두 개의 FC layer에만 dropout을 적용하였습니다. dropout을 통해 overfitting을 피할 수 있었고, 수렴하는데 필요한 반복수는 두 배 증가되었다고 나와있습니다.  

---

## **Detalis of learning**  

이번 section은 학습시킬 때 설정하였던 hyper parameter에 대한 내용입니다.

AlexNet은 momentum=0.9, batch size=128, weight decay=0.005로 설정한 SGD(stochastic gradient descent)를 이용했습니다. 또 weight decay의 중요성을 강조합니다. weight decay는 training error를 낮출 수 있다고 말합니다. 아래 수식은 AlexNet에서 이용한 weight 갱신 방법입니다.  

![9번사진](https://user-images.githubusercontent.com/83005178/170996609-d9a6da7f-4807-483c-b92c-4b74a10979e3.png)  
 
weight 초기화는 평균 0, 분산 0.01인 가우시안 분포를 이용했으며
bias 초기화는 Conv2,4,5와 FC layer에서는 1로, 나머지 layer에서는 0으로 초기화 시켰습니다.  

learning late는 0.01로 초기화 시켰고 validation error가 상향되지 않으면 10으로 나눠주었습니다. 학습 종료 전에 3번의 learning late 감소가 이루어 졌습니다.

---

# **References**

[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

[AlexNet](https://naknaklee.github.io/classification/2020/04/22/AlexNet-Post-Review/)
