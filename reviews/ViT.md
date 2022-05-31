---
title: "Vision Transformer Review"
author: "하영수"
---

# Vision Transformer(ViT)  

![vit현재](https://user-images.githubusercontent.com/83005178/171037619-6cfc55c6-4f55-4bc5-b4d0-400009d9ef80.PNG)  

**An Image is Worth 16*16 Words: Transformers for Image Recognition at Scale, ICLR 2021** 논문의 **ViT**는 2022년 5월 기준 이미지넷 벤치마크 TOP 10중 다수를 차지하는 모델입니다.  **NLP**에서는 지배적인 standard architecture였으나, vision 분야에서의 응용은 제한적이었던 **Transformer**가 어떻게 자리잡았는지에 대하여 정리해 보겠습니다.  

---

### Background  


**Transformer**는 NLP에서는 지배적인 standard architecture가 되었으나 vision분야에서의 활용은 제한적이었습니다.  
vision 분야에서 attention을 적용한다고 해도, convolutional 네트워크와 혼합되어 사용되거나, convnet의 전반적인 틀은 남겨둔채 몇몇 요소들을 대체하는 식으로 사용되어 왔습니다.  

### Architecture  

![vit구조](https://user-images.githubusercontent.com/83005178/171037653-14638364-395d-4099-8e74-a783bdf767e6.png)

논문 저자들은 표준 트랜스포머를 아주 약간의 수정 후 이미지에 직접 적용하는 실험을 했습니다. 이미지를 작은 patch들로 분할하고 이 patch들의 linear embedding의 sequence를 트랜스포머의 인풋으로 전달합니다.  즉, 하나 하나의 patch를 NLP의 token으로 간주하는 것입니다. 이후 MLP를 Classification head로써 트랜스포머 인코더 위에 추가하여 분류 작업을 수행합니다.  

### Experiment  

![vit3모델](https://user-images.githubusercontent.com/83005178/171038522-9ddea224-9777-482e-b073-c61e7938e0a4.png)  

ViT의 3가지 variant

![vit비교](https://user-images.githubusercontent.com/83005178/171038536-82ebcf64-e363-4c02-b7fa-2000cd0276b4.png)  

타 SOTA(State-of-the-art) 모델들과의 비교

위 사진을 보면 JFT + pre-train된 ViT-H/14, ViT-L/16가 기존의 SOTA를 능가하면서, 리소스는 더 낮은 것을 확인할 수 있습니다. 즉 충분히 큰 데이터에 사전훈련된 이후 더 작은 데이터 셋에 대해 전이되어 학습이 이루어 질 때 기존 SOTA모델들의 성능에 필적하거나 혹은 능가하는 결과를 낸다는 것입니다.  

![사전학습1](https://user-images.githubusercontent.com/83005178/171040402-8797e8be-2382-4ce4-86ef-12389dc9a87a.jpg)  

![사전학습2](https://user-images.githubusercontent.com/83005178/171040412-f711cea9-8b7c-4c9b-9faf-3b139b8cd303.jpg)  

ViT는 inductive bias가 기존의 ResNet보다 훨씬 더 적음에도 불구하고 훨씬 뛰어난 성능을 보입니다.  

![가성비](https://user-images.githubusercontent.com/83005178/171040744-c358dea9-1acd-4e4f-be64-2ed4ed72b94a.jpg)  

모든 ViT 모델이 ResNet 모델들을 성능-계산 비용의 trade-off에서 ResNet을 압도하는 모습을 볼 수 있습니다. 또한 동일한 성능 대비 계산 비용이 2~4배가 적은 모습을 보입니다.  



### Conclusion  

- Computer Vision에서 self attention을 사용한 이전 연구들과 달리 inductive bias를 강제로 추가하지 않고도 SOTA를 달성하였습니다.   
- 거대한 데이터셋에 대해 pre-training이 이루어져야만 좋은 성능을 줄 수 있습니다.  
- large dataset에 대해 사전 훈련을 할 때 굉장히 성능이 좋고, 비용이 상대적으로 저렴합니다.  

---

### References  

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  

[Seq2Seq](https://wikidocs.net/24996)  

[Attention Mechanism](https://wikidocs.net/48948)  

[Tokenization](https://wikidocs.net/21698)