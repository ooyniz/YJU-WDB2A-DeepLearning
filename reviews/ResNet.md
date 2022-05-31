---
title: "ResNet Review"
author: "하영수"
---

# ResNet

**Deep Residual Learning for Image Recognition** 논문의 **ResNet**은 2015년도에 ILSVRC에서 우승을 하였고, 총 152개의 레이어를 가진 매우 깊은 네트워크 모델입니다.  
이렇게 깊은 네트워크를 구성하게 된다면 많은 문제점이 존재합니다. ResNet에서 이 문제점들을 어떻게 해결 하였는지에 대해 정리해 보겠습니다.  

---

### Problem of Plain Network  

**Plain Network**이란 **AlexNet**과 **VGGNet**처럼 **skip/shortcut connection**을 사용하지 않은 네트워크들 입니다. 이러한 네트워크들은 **layer**의 깊이가 점점 깊어질 수록 문제점이 있는데 바로 **gradient vainishing**(기울기 소실)과 **gradient exploding**(기울기 폭발)입니다.  

**Neural Network**에서는 기울기를 구하기 위해서 가중치에 해당하는 손실 함수의 미분을 오차 역전파(Back Propagation)을 통해서 구하게 됩니다.  
이 과정에서 활성화 함수(Activation Function)의 편미분을 구하고 그 값을 곱해줍니다. 이것은 layer가 뒷단으로 갈수록 활성화 함수의 미분값이 0으로 수렴하거나 매우 큰 값으로 발산하게 됩니다.  

이렇게 신경망이 깊어질 때, 작은 미분 값이 여러번 곱해지게 되면 0에 가까워 지는 현상이 Gradient Vanishing(기울기 소실)입니다.  
그 반대로 기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산되기도 하는데, 이를 Gradient Exploding(기울기 폭발)이라고 합니다.  

![resnet1](https://user-images.githubusercontent.com/83005178/171004697-e97ac13e-a167-4425-8260-b519be9063f9.png)  

더 깊은 56-layer의 network의 error rate가 20-layer의 error rate 보다 높은 이유가 gradeint vanishing 때문입니다.  

해당 논문에서는 layer를 깊게 쌓되 많은 layer들이 identity mapping으로 이루어지도록 만든다면 깊은 모델이 얕은 모델에 비해 높은 training error를 가지면 안되는 것이 gradient vanishing problem을 해결하기 위한 핵심이라고 했습니다.  

---

### ResNet Learning  

![기존의학습](https://user-images.githubusercontent.com/83005178/171005362-2a001d6f-9390-41ec-a529-201ce70081a3.png)  

위 사진은 기존의 Neural Network의 underlying mapping으로, H(x)라면, 즉 H(x)를 최소시키는 것이 학습의 목표였다면 해당 논문에서의 목표는 F(x) = H(x) - x 를 최소화시키는 Residual mapping으로 H(x)를 재정의 하였습니다.  

![새로운학습](https://user-images.githubusercontent.com/83005178/171005561-eb8ca632-9225-4703-87b3-76a2f04a71d2.png)  

새로운 방식은 skip connection에 의해 출력값에 x를 더하고 H(x) = F(x) + x로 정의합니다. 그리고 F(x) = 0이 되도록 학습하여 H(x) = 0 + x가 되도록 합니다. 이 방법이 최적화하기 훨씬 쉽다고 합니다. 미분을 했을 때 더해진 x가 1이 되어 기울기 소실 문제가 해결됩니다.  이렇게 gradient vanishing problem을 해결하게 된다면 정확도가 감소되지 않고 신경망의 layer를 더욱 더 깊이 쌓을 수 있어 더 나은 성능의 신경망을 구축할 수 있게됩니다.  

---

### Architecture  

![아키텍처](https://user-images.githubusercontent.com/83005178/171005919-7410143d-3ebb-45f1-9551-40edb3c5fac6.png)  

왼쪽 구조는 VGG-19 입니다.  
중간 구조는 VGG-19가 더 깊어진 34-layer plain network입니다.  
오른쪽 구조는 34-layer residual network(ResNet)이며 plain network기반의 skip/short connection이 추가되었습니다.  

---

### Plain Nerwork VS ResNet  

![비교](https://user-images.githubusercontent.com/83005178/171006472-fcef7dfd-4309-49e6-a9bc-34be2ce343af.png)
![비교2](https://user-images.githubusercontent.com/83005178/171006478-4d22b0ad-2715-4b8e-9351-5faf8c60e91b.png)

plain network에서 34-layer의 성능보다 18-layer의 성능이 좋습니다.  
이것이 앞서 언급한 Gradient Vanishing Problem때문입니다.   

ResNet에서는 Gradient Vanishing Problem가 skip connection에 의해 해결되어 34-layer의 성능이 18-layer보다 뛰어납니다.  

---  

### References  

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

[VGP](https://wikidocs.net/61375)  