---
title: "LeNet-5 Review"
author: "유원규"
---

# LeNet-5  


## 등장배경

Yann LeCun이 손으로 적힌 우편번호를 전통적인 방법보다 효율적으로 확인하기 위해 고안된 CNN구조를 말하는 것이다.  패턴 인식에서 이용되는 전통적인 모델은 hand-designed feature extractor(손으로디자인한 특징추출기)로 특징을 추출하고 fully-connected multi-layer networks를 분류기로 사용하기에 여러 가지 문제점이 발생한다.  

<br>

---

## 전통적인 방법의 문제점  

1. **Hand-designed feature extractor는 제한된 특징만 추출합니다.**
hand-designed feature extractor는 입력으로부터 관련있는 정보만 수집하고 무관한 정보를 제거합니다. 이는 사람이 설계한 feature extrator이 추출한 정보만 분류기로 전달되므로 제한된 학습이 이루어질 수 밖에 없습니다. 따라서 좋은 학습은 feature extractor 그 자체에서 학습이 이루어져야 한다고 말합니다.  

<br>


2. **너무 많은 매개변수를 포함합니다.**
하나의 이미지는 몇 백개의 변수(pixel)를 포함하고 있습니다. 또한 fully-connected multi layer의 첫 번째 layer에 이미 몇 만개의 가중치를 포함합니다. 이러한 매개변수는 시스템의 capacity를 증가시키므로 더 많은 훈련셋이 필요하게 됩니다. 또한 많은 가중치를 저장해야 하므로 메모리에 저장공간이 많이 필요하게 됩니다.  

<br>

3. **입력값의 topology가 완전히 무시됩니다.**
이미지는 2D 구조를 갖고 있으므로 인접한 변수(pixel)들은 공간적으로 매우 큰 상관관계가 있습니다. fully-connected multi layer(완전히 연결된 다층)는 인접한 변수들을 단순히 배열하여 학습하므로 공간적인 정보를 이용하지 못합니다.  

<br>

---

## 문자 인식 업무에서 CNN  

CNN은 약간의 shift, scale, distorition 불변성을 갖기 위해 세 개의 아이디어를 결합했습니다. (Local receptive field, Shared-weight, sub-sampling)  

1. **수용 영역 – receptive field**  

![1번사진](https://user-images.githubusercontent.com/83005178/170997449-469797ee-ef14-4ffe-b34e-74d61b89fc18.png)


CNN은 hidden unit의 receptive field를 local로 제한함으로써 local featrue를 추출합니다.  
하나의 layer는 이전 layer의 제한된 지역에 위치해 있는 유닛의 집합을 일력으로 취합니다.  
receptive field를 이용하여 conner, edge, end-point와 같은 특징을 추출할 수 있게 됩니다.  
추출된 특징들은 고차원의 특징을 검출 하기 위해 그 다음 layer에서 결합됩니다.
이를 통해 shift, distorition이 발생하더라도 특징을 나타내는 배열이 receptive field에 검출 된다면, 해당 특징을 반영한 feature map을 만들어 낼 수 있습니다.

또한 아래 그림처럼 receptive field를 이용하면 parameter 수를 줄일 수 있게 됩니다.  

![2번사진](https://user-images.githubusercontent.com/83005178/170997459-7eff7807-62bc-436f-879c-e73656f15254.png)  

<br>

2. **가중치 공유 - shared weight**  

CNN은 가중치 배열을 강제로 복제함으로써 자동으로 shift 불변성을 얻습니다.  

feature map에 있는 unit은 동일한 weights와 bias를 공유합니다. 공유된 weight 집합을 convolution kernel로 이용하여 입력에서 모든 위치에서 동일한 특징을 추출합니다. 예를 들어, 5x5 kernel은 5x5사이즈와 설정된 Stride에 맞춰 feature map를 돌아다니며 계산하지만, 5x5의 weight와 1개의 bias만 back propagation으로 학습을 합니다.  

weight를 공유하게 되면 학습 파라미터가 느는 것이 아니라, kernel를 총 몇개로 설정하는가에 따라 output인 feature map의 수와 학습해야하는 parameter만 늘게됩니다. 이 기법을 사용하면 요구되는 계산 capacity를 줄여주고, 학습할 parameter의 수를 줄여줌으로써 Overfitting를 방지하게 되어 test error와 training error 사이의 gap도 줄여줍니다. 실제로 LeNet-5에는 340,908 connection이 존재하지만 60,000개의 trainable parameter만 존재하게 됩니다.  

또한 이 기법은 입력 이미지가 변환됬으면 feature map의 결과값도 동일한 양만큼 변화됩니다. 이 덕분에 CNN은 입력의 왜곡과 변환에 대한 Robust를 갖게 됩니다.  

<br>

3. **sub-sampling**

sub-sampling은 현대의 pooling을 의미합니다. LeNet-5에서는 average pooling을 이용합니다.  

논문에서 한번 특징이 검출되면 위치 정보의 중요성이 떨어진다고 말합니다. 예를 들어, 입력 이미지가 7 이면 좌측 상당에 수평적인 end-point, 우측 상단에 corner, 이미지의 아래 부분에 수직적인 end-point를 포함합니다. 이러한 각 특징의 위치 정보는 패턴을 식별하는 것과 무관할 뿐만 아니라, 입력값에 따라 특징이 나타나는 위치가 다를 가능성이 높기 때문에 잠재적으로 유해하다고 말합니다.  

feature map으로 encoding 되는 특징들의 위치에 대한 정확도를 감소시키기 위한 가장 간단한 방법은 feature map의 해상도를 감소시키는 것이라고 말합니다. sub-sampling layer에서 local average와 sub-sampling을 수행하여 feature map의 해상도를 감소시키고 distortion과 shift에 대한 민감도를 감소시킬 수 있다고 말합니다.  

또 위치 정보를 소실시키면서 생기는 손실은, feature map size가 작아질수록 더 많은 filter를 사용하여 다양한 feature를 추출하여 상호보완할 수 있도록 합니다.  

---  

## LeNet-5 Architecture  


LeNet-5는 32x32 크기의 흑백 이미지에서 학습된 7 layer Convolutional Neural Network 입니다.  
[Conv(C1) - Subsampling(S2) - Conv(C3) - Subsampling(S4) - Conv(C5) - FC – FC]  

![3번사진](https://user-images.githubusercontent.com/83005178/170998150-4b27813f-07de-47bb-9a49-dd2fb881dee4.png)  

Cx : convolution layer  
Sx : subsampling (pooling) layer  
Fx : fully-connected layer  
x : index of the layer  
Input  

입력 이미지는 32x32 입니다. 실제 문자 이미지는 28x28 영역에서 20x20 크기의 pixel이 중앙에 있습니다. 실제 문자 이미지보다 큰 이유는 receptive field의 중앙 부분에 corner 또는 edge와 같은 특징들이 나타나길 원하기 때문입니다.  

개인적인 생각으로는 현대의 padding 기법을 사용하는 이유와 비슷하다고 생각합니다.  

- **Layer C1**
5x5 크기의 kernel 6개와 stride=1 을 지닌 convolutional layer 입니다.
입력 크기는 32x32x1 이고, 출력 크기는 28x28x6 입니다.
156개의 trainable parameters와 122,304개의 connections를 갖고 있습니다.  
<br>

- **Layer S2**
2x2 크기의 kernel 6개와 stride=2 을 지닌 subsampling layer 입니다.
입력 크기는 28x28x6 이고, 출력 크기는 14x14x6 입니다.
12개의 trainable parameters와 5880개의 connections를 갖고 있습니다.  
<br>

- **Layer C3**
5x5 크기의 kernel 16개와 stride=1 을 지닌 convolution layer 입니다.
입력 크기는 14x14x6 이고, 출력 크기는 10x10x16 입니다.
1,516개 trainable parameters와 151,600 connections를 갖고 있습니다. 
아래 표는 C3의 feature map과 연결된 S2의 feature map을 보여줍니다.  

![4번사진](https://user-images.githubusercontent.com/83005178/170998439-ec242e11-cd57-40a0-8f3c-51502be1745f.png)


표를 보면 모든 S2의 feature map이 C3의 feature map에 연결되지 않았습니다.
그 이유는 두가지로 설명됩니다.  
1. 모든 feature map을 연결하지 않기 때문에 connetion의 숫자를 제한시킵니다.
2. 서로 다른입력값을 취하므로 C3의 각 feature map은 서로 다른 feature를 추출(상호보완적으로)하도록 합니다.  
<br>

- **Layer S4**
2x2 크기의 kernel 16개와 stride=2 을 지닌 subsampling layer 입니다.
입력 크기는 10x10x16 이고, 출력 크기는 5x5x16 입니다.
32개의 trainable parameters와 2,000개의 connections를 갖고 있습니다.  
<br>

- **Layer C5**
5x5 크기의 kernel 120개와 stride=1 을 지닌 convolutional layer 입니다.
입력 크기는 5x5x16 이고, 출력 크기는 1x1x120 입니다.
10,164개의 trainable parameters를 갖고 있습니다.  
<br>

- **Layer F6**
tanh 함수를 활성화 함수로 이용하는 fully-connected layer 입니다.
입력 유닛은 120개 이고, 출력 유닛은 84개 입니다. 
출력 유닛이 84인 이유는 아래 보이는 ASCII set을 해석하기 적합한 형태로 나와주길 바라는 마음으로 설정했다고 말합니다. 각각의 문자가 7x12 크기의 bitmap 이기 때문입니다.  
![5번사진](https://user-images.githubusercontent.com/83005178/170998601-7e105d14-3316-480a-930e-915f70450180.png)  

<br>


- **Layer F7**
RBF(Euclidean Radia Basis Function unit)를 활성화 함수로 이용하는 output layer 입니다.
입력 크기는 84 이고, 출력 크기는 10 입니다.
MNIST 데이터를 이용했기 때문에 출력크기가 10 입니다.  
<br>

- **Loss function**
Loss function은 MSE(평균 제곱 오차)를 이용했습니다.  
<br>

---

# **References**

[Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

[LeNet-5](https://arclab.tistory.com/150)

