# YJU-WDB2A-DeepLearning
## :mag_right: 영진전문대학교 WDB2A반 1조 딥러닝 스터디

## WITH

<table>
  <tr> 
    <td align="center"><a href=https://github.com/Hannah0su><img src="https://user-images.githubusercontent.com/102000749/165738552-60e1eac0-3c50-4568-ae38-767c44b3b018.jpg" width="100px;" alt=""/><br /><sub><b>Ha Youngsu</b></sub></a><br /><a href="https://hannah0su.github.io/" title="Code">🏠</a></td>
    <td align="center"><a href=https://github.com/baegjhoon><img src="https://user-images.githubusercontent.com/102000749/165739357-9ea66cf1-8a6e-4b9a-bf77-0a8c9e1a465a.png" width="100px;" alt=""/><br /><sub><b>Baeg Junghun</b></sub></a><br />🏠</td>
    <td align="center"><a href=https://github.com/sila0319><img src="https://user-images.githubusercontent.com/102000749/165739259-24741b3b-92d2-49df-8496-7dab8f58bd97.png" width="100px;" alt=""/><br /><sub><b>Ryu wonkyu</b></sub></a><br />🏠</td>
  </tr>
</table>

## 🔍 Content
1. [🖥 인공지능 알고리즘](#-인공지능-알고리즘)  
2. [🧠 딥러닝(Deep Learning) 이란?](#-딥러닝deep-learning-이란)
3. [📚 The History of Deep Learning](#-the-history-of-deep-learning)
   + [1세대: Perceptron](#1세대-perceptron)
   + [2세대: Multilayer Perceptron](#2세대-multilayer-perceptron)
   + [3세대: Unsupervised Learning - Boltzmann Machine](#3세대-unsupervised-learning---boltzmann-machine)
4. [🐈 CNN (Convolutional Neural Network, 컨볼루션 신경망)](#-cnn-convolutional-neural-network-컨볼루션-신경망)
5. 📝 **reviews**
   + [LeNet-5](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/LeNet-5.md)
   + [AlexNet](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/AlexNet.md)
   + [VGGNet](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/VGGNet.md)
   + [ResNet](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/ResNet.md)
   + [ViT](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/ViT.md)
6. 🛠 **implementation**
   + [ViT](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/WD2A_1%EC%A1%B0_ViT_%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0.ipynb)
   + [ViT-2](https://github.com/ooyniz/YJU-WDB2A-DeepLearning/blob/main/reviews/WD2A_1%EC%A1%B0_ViT_%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B02.ipynb)

<br>
<br>

## 📌 목표
- 인공지능과 딥러닝의 개념을 **이해**합니다.
- 딥러닝 구현과 학습을 위한 데이터를 이해하고 데이터를 준비하는 방법을 **학습**합니다.

<br>

### 🔗 함께 볼 것
* [모두를 위한 딥러닝 강좌 시즌1](https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
* [딥러닝 홀로서기 - Idea Factory KAIST](https://www.youtube.com/watch?v=hPXeVHdIdmw&list=PLSAJwo7mw8jn8iaXwT4MqLbZnS-LJwnBd)
  * 입문용, 딥러닝 이해
* [논문으로 시작하는 딥러닝](https://www.edwith.org/deeplearningchoi)

<br>

## 🖥 인공지능 알고리즘

인공지능은 사고나 학습 등 인간이 가진 능력을 컴퓨터를 통해 구현하는 기술입니다. 

인공지능은 일반 컴퓨터의 처리 방식과는 다르게, 

사람이 원하는 결과 데이터를 제공하면 인공지능이 알아서 처리 방법을 만들어 사람에게 처리 결과를 보여줍니다.

![image](https://user-images.githubusercontent.com/102000749/170696847-2aef033e-0c94-4498-ae44-2517a8581330.png)

**<컴퓨터가 처리하는 방식>**

![image](https://user-images.githubusercontent.com/102000749/170696882-18329f75-60b7-41cd-9632-c6ab9feb4df5.png)

**<인공지능이 처리하는 방식>**

<br>


그리고 인공지능에는 머신러닝과 딥러닝이 있습니다. 

머신러닝과 딥러닝의 가장 큰 차이점은 **학습할 데이터**에 있습니다. 

**머신러닝**`Machine Learning`은 학습에 필요한 feature를 **사람이 직접 제공**해야 하지만, 

**딥러닝**`Deep Learning`은 **스스로** feature를 추출해서 데이터 학습에 적용할 수 있습니다. 

<br>

## 🧠 딥러닝(Deep Learning) 이란?

**딥 러닝**`Deep Learning` 은 인간의 신경망의 원리를 모방한 심층신경망 이론을 기반해 고안된 머신러닝 방법의 일종입니다.

특히 머신 비전 분야에서 신경망은 일반적으로 데이터와 예제 데이터에 대한 사전 정의된 결과와 같은 **지도 학습**을 통해 학습됩니다.

<br>

딥 러닝이 기존의 통계학이나 다른 머신러닝 방법과 다른 큰 차이점은 인간의 뇌를 기초로하여 설계되었다는 점입니다.

인간은 컴퓨터가 아주 짧은 시간에 할 수 있는 계산도 쉽게 해낼 수 없는 반면, 컴퓨터는 인간이 쉽게 인지하는 사진이나 음성을 해석하지 못합니다.

이는 인간의 뇌가 엄청난 수의 **뉴런**`Neuron`과 **시냅스**`Synapse`로 이루어져 있기 때문입니다.

각각의 뉴런은 큰 연산을 수행하는 능력이 없지만 수많은 뉴런들이 복잡하게 연결되어 병렬연산을 수행해 컴퓨터가 하지 못하는 음성, 영상인식을 수월하게 할 수 있습니다.

딥 러닝은 이 뉴런과 시냅스의 병렬연산을 컴퓨터로 재현하는 방법이며, 기존의 로직과는 많은 차이를 보입니다.

<br>

딥러닝의 또 다른 이름은 **심층신경망(Deep Neural Network, DNN)** 입니다. 

이름에서 알 수 있듯이 딥러닝은 뉴런으로 구성된 레이어를 여러 개 연결해서 구성한 네트워크이며, 

네트워크를 어떻게 구성하느냐에 따라 **통 NN**`Neural Network`으로 불립니다. 

대표적인 예로 **CNN**`Convolutional Neural Network`, **RNN**`Recurrent Neural Network` 등이 있습니다. 
  
<br>
  
------


# 📚 The History of Deep Learning

딥러닝의 역사는 크게 3가지 세대로 나누며, 1세대는 최초의 인경신공망인 퍼셉트론, 2세대는 다층 퍼셉트론, 마지막 3세대를 현재의 딥러닝이라고 할 수 있습니다.

## 1세대: Perceptron

![image](https://user-images.githubusercontent.com/102000749/172045967-d05fd200-8b8c-4828-87b3-c7f7809050a0.png)


**[인공신경망](https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B3%B5_%EC%8B%A0%EA%B2%BD%EB%A7%9D)**`Neural Network`의 기원이 되는 퍼셉트론은 1958년 [Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt)가 제안하였는데, n개의 입력을 받아 특정한 연산을 거쳐 하나의 값을 출력하는 방식입니다.

이 연산은 1차함수의 형태`f(x) = w * x + b`를 띄며, 여기에 **활성화 함수**`Activation Function`을 적용하여 최종 값을 출력합니다.  

이 출력값은 참인지 거짓인지를 판단합니다. (0 vs 1)

<br>

n개의 input과 1개의 output에 대하여 각각의 input의 weight를 wi라 한 후 퍼셉트론을 수식으로 나타내면 다음과 같습니다.

![image](https://user-images.githubusercontent.com/102000749/170698693-9585ff6f-2796-4f4d-b490-ae8ba0f6a8aa.png)

![image](https://user-images.githubusercontent.com/102000749/170698749-81713f87-fa86-45b8-8e74-c674ce5c1b11.png)

즉, n개의 input의 **선형결합**`Linear Combination`에 Activation 함수를 적용하여 0$$1 사이의 확률값으로 y값을 제공합니다. 

이것이 **인공신경망 모형**의 시작입니다. 

허나 이 모형은 아주 간단한 XOR problem마저 학습하지 못하는 등, 심각한 문제가 있는데(아래 그림) 

이 때문에 한동안 발전없이 포류되게 됩니다.

![image](https://user-images.githubusercontent.com/102000749/170698860-90953ac1-6603-4626-8852-c0d00420618d.png)

## 2세대: Multilayer Perceptron

XOR 같은 간단한 것도 학습하지 못하는 퍼셉트론의 단점을 해결하기 위한 방법은 의외로 단순하였는데 

**입력층**`Input layer`와 **출력층**`output layer`사이에 하나 이상의 **은닉층**`hidden layer`를 추가하여 학습하는 것, **[다층 퍼셉트론](https://en.wikipedia.org/wiki/Multilayer_perceptron)**`Multilayer perceptron`이라 합니다.  

아래의 그림은 은닉층의 개수에 따라 모델의 분류력이 좋아진다는 것을 보여줍니다.  

![image](https://user-images.githubusercontent.com/102000749/170699209-d368e1e3-cc65-4639-a8e4-8b39c28d2f17.png)

하지만 이런 방법은 은닉층의 개수가 증가할수록 가중치의 개수도 증가해 학습이 어렵다는 단점이 존재했습니다. 

이러한 문제를 **[에러역전파알고리즘](https://en.wikipedia.org/wiki/Backpropagation)**`Backpropagation Algorithm`을 개발하여 다층 퍼셉트론의 학습을 가능하게 하였습니다. 

에러역전파알고리즘으로 다층퍼셉트론을 학습할 수 있게 되었으나 이것을 실제로 사람들이 이용하기에는 많은 어려움이 따랐는데 그 이유들은 다음과 같습니다.

1. 수많은 **레이블**`Label`이 필요하다.
2. **과적합**`Overfitting`으로 인해 성능이 떨어질 수 있다.
3. **기울기 소실**`Gradient Vanishing` 문제가 발생할 수 있다.
4. **로컬 미니멈**`Local minimum`에 빠질 가능성이 있다.

추정해야 하는 모수가 많기 때문에 데이터가 많이 필요하고 그 중에서도 [labeled data](https://en.wikipedia.org/wiki/Labeled_data)가 많이 필요합니다. 

허나 우리가 갖고 있는 데이터는 unlabeled data가 훨씬 많으며 실제 인간의 뇌의 학습 중 많은 부분이 unlabeled data를 이용한 Unsupervised Learning이며, 

적은 양의 labeled data로 다층퍼셉트론을 학습하면 종종 hidden layer가 1개인 경우보다 성능이 떨어지는 경우를 관찰할 수 있으며 이것이 **[과적합](https://ko.wikipedia.org/wiki/%EA%B3%BC%EC%A0%81%ED%95%A9)**`Overfitting`의 예시입니다.

다음으로 Activation function을 살펴보면 logistic function이든 tanh function이든 가운데 부분보다 양 끝이 현저히 기울기의 변화가 작은 것을 발견할 수 있습니다.

![image](https://user-images.githubusercontent.com/102000749/170700277-299d32eb-6971-4b82-85dd-ede9b132faba.png)

때문에 학습이 진행될수록 급속도로 기울기가 0에 가까워져서 나중에는 거의 [Gradient descent](https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95)가 일어나지 않아 학습이 되지 않는 단점이 있습니다.

마지막으로 최소제곱추정량이나 최대가능도추정량등 직접적으로 최소값을 구하는 방법을 이용하지 못하고 알고리즘을 이용하여 최소값에 가까워지게 했기 때문에

학습에서 나온 최소값이 과연 **진짜 최소값**`Global minima`인가? **국소 최소값**`Local minima`는 아닌가..에 대한 의문점이 풀리지 않게 됩니다. 

시작점을 어떻게 두느냐에 따라 **Local minima**에 빠질 수도 있기 때문입니다.

![image](https://user-images.githubusercontent.com/102000749/170700310-5241bf59-3733-44aa-aa2a-2886191f76bb.png)

이런 문제점들 때문에 실제로 Neural Network은 **지지벡터머신**`Support Vector Machine`등에 밀려 2000년 초까지 제대로 활용되지 못하였습니다.

## 3세대: Unsupervised Learning - Boltzmann Machine

다층 퍼셉트론의 단점으로 인해 인공신경망 이론이 잘 활용되지 못하던 와중, 2006년 볼츠만 머신을 이용한 학습방법으로 인해 딥 러닝이 다시 재조명되기 시작합니다.

**볼츠만 머신**`Boltzmann Machine`의 핵심 이론은 바로 **비지도학습**`Unsupervised Learning`, 즉 **레이블 데이터 없이 미리 충분히 학습을 한 이후 기존의 지도학습을 수행**하는 것입니다.

아래 그림에 대략적인 묘사가 표현되어 있는데 아기들은 단어나 음, 문장의 뜻을 전혀 모르는 상태로 학습을 시작하게 되고 

**음소**`phoneme`, **단어**`word`, **문장**`sentence`순으로 비지도학습을 수행하게 되며 그 후에 정답을 가지고 지도학습을 수행하게 됩니다.

![image](https://user-images.githubusercontent.com/102000749/170700914-84beb7cd-dbc1-4464-8fe3-7d07133d54c0.png)

이러한 방법으로 앞서 말한 기울기 소실, 과적합 문제의 극복이 가능하게 되었으며, 지도학습 이전의 **비지도 사전학습**`Unsupervised pre-train`을 통한 올바른 초기값 선정으로 로컬 미니멈 문제도 해결이 가능하게 되었습니다.
 

<br>

-------

## 🐈 CNN (Convolutional Neural Network, 컨볼루션 신경망)

![image](https://user-images.githubusercontent.com/102000749/170703624-83c024a1-420c-4b61-8bd0-fc90fcefd075.png)

**CNN**`Convolutional Neural Network`는 **합성곱신경망**으로도 불립니다. 

주로 **시각적 이미지**를 분석하는 데 사용되는데 머신러닝의 한 유형인 **딥러닝**에서 가장 많이 사용되고 있는 알고리즘입니다.

초창기 CNN을 개발한 사람들은 **고양이의 시선**에 따라 뇌에서 자극 받는 위치가 모두 다르다는 점을 착안하여 CNN의 아이디어를 얻었습니다. 

CNN은 이미지 전체를 작은 단위로 쪼개어 각 부분을 분석하는 것이 핵심입니다.

![image](https://user-images.githubusercontent.com/102000749/170703326-d881a95e-78e3-4f1d-879d-e4b3bdfa5dad.png)


CNN은 이미지를 인식하기 위해 **패턴**을 찾는 데 유용합니다. 

데이터를 통해 특징을 스스로 학습하고, 패턴을 사용하여 이미지를 분류하고 특징을 수동으로 추출할 필요가 없습니다. 

또한 기존 네트워크를 바탕으로 새로운 인식 작업을 위해 CNN을 재학습하여 사용하는 것이 가능합니다.

CNN은 **이미지 인식**이 주로 사용되는 휴대폰 잠금해제 인식이나 자율 주행 자동차와 같은 분야에 많이 사용됩니다. 

응용 분야에 따라 CNN을 처음부터 만들 수도 있고, 데이터셋으로 사전 학습된 모델을 사용할 수도 있습니다.

CNN은 다른 신경망과 마찬가지로 입력 계층, 출력 계층 및 두 계층 사이의 여러 은닉 계층으로 구성됩니다.

![image](https://user-images.githubusercontent.com/102000749/170702307-081b9559-ba5d-4520-b36c-3f42f750d826.png)

각 계층은 해당 데이터만이 갖는 특징을 학습하기 위해 데이터를 변경하는 계산을 수행합니다. 

가장 자주 사용되는 계층으로는 컨볼루션, 활성화/ReLU, 풀링이 있습니다.

![image](https://user-images.githubusercontent.com/102000749/170702948-254655db-fa65-4ea6-abd2-c75e876c773f.png)

>컨볼루션
>>각 이미지에서 특정 특징을 활성화하는 컨볼루션 필터 집합에 입력 이미지를 통과시킵니다.

>ReLU(Rectified Linear Unit)
>> 음수 값을 0에 매핑하고 양수 값을 유지하여 더 빠르고 효과적인 학습을 가능하게 합니다. <br> 이때 활성화된 특징만 다음 계층으로 전달되기 때문에 이 과정을 활성화라 부르기도 합니다.

>풀링
>> 비선형 다운샘플링을 수행하고 네트워크에서 학습해야 하는 매개 변수 수를 줄여서 출력을 간소화합니다.

이러한 작업이 수십 개 또는 수백 개의 계층에서 반복되어 각 계층이 여러 특징을 검출하는 방법을 학습하게 됩니다.

![image](https://user-images.githubusercontent.com/102000749/170702884-d8a73952-9ee7-4d5a-8eec-c8ed8fc9e7f7.png)

