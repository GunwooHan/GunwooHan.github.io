---
layout: post
title:  "03.DCGAN"
date:   2021-01-04 23:42:29 +0900
categories: GAN_In_Action
---

# DCGAN
---
DCGAN은 deep convolutional generative adversarial network의 줄임말로, 기존에 fcn 레이어를 사용하던 GAN에서 conv 레이어로 교체한 모델이다. 그래서 모델 구조는 거의 유사하지만 디테일한 부분에서 차이가 있다.  

1. 판별자에서 pooling 레이어를 모두 strided conv 레이어로 바꾸고, 생성자는 pooling 레이어를 transposed conv 레이어로 바꿈
2. 생성자의 초기실패를 막기 위해 batch norm을 사용하지만, 모든 레이어에 적용하면 sample oscillation과 model instability가 발생하여 생성자 output, 판별자 input 레이어에는 적용하지 않음
3. fcn 레이어 삭제
4. 생성자 활성화 함수는 relu를 쓰되 마지막 결과에서는 tanh를 사용
5. 판별자 활성화 함수는 leaky relu 사용
