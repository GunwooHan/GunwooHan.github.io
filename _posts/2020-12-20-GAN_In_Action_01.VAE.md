---
layout: post
title:  "01.VAE(변이형 오토인코더)"
date:   2020-12-20 01:49:29 +0900
categories: GAN_In_Action
---

# VAE
- - -

VAE(Variational AutoEncoder)는 데이터의 확률 분포를 학습하기 위해 인코더와 디코더, 2개의 모델로 구성되어 있다. 이때 인코더는 데이터를 추상화하고, 디코더는 추상화된 latent vector를 원본 데이터처럼 복원하는 역할을 하게 된다.  
<br>

VAE 3가지 단계로 구성된다.

<br>

> 1단계(녹색)   : 압축  
> 2단계(빨간색) : 잠재공간 매핑  
> 3단계(파란색) : 압축 해제  

![image](/public/img/vae-gaussian.png)

위 그림에서 같이 인코더가 데이터 x를 받아 평균과 표준편차를 출력하는데, 이 평균과 표준편차로 정규분포를 만든다. 디코더는 인코더가 만들어낸 z값 으로 원본데이터를 복원한다.  

<br>
사실 책에서도 이론보다는 실습코드가 주라 대충 컨셉만 이해하고 바로 코드로 가보자


## 코드(Keras)
- - - 
### 라이브러리 불러오기 
{% highlight python %}
from tensorflow.keras.layers import Input,Dense,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
{% endhighlight %}

<br>
책에서는 tensorflow 2 기준으로 쓰여져 있어 TF2로 진행했다.  
기본적으로 필요한 layer, metric을 호출해주고, 데이터셋은 MNIST로 진행했다.
<br>


### 하이퍼 파라미터 선언
{% highlight python %}
batch_size = 100
original_dim =784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.
{% endhighlight %}

코드 실습에서 쓰일 batch_size와 input_shape 같은 하이퍼 파라미터를 재사용 할 수 있도록 변수 형태로 선언해준다
<br>
이미지 x를 넣어서 잠재변수(latent variable) z를 추출하는 인코더 선언 부분  
> 입력  : x (이미지 데이터)  
> 출력  : z, z 평균, z 표준편차
<br>
{% highlight python %}
x = Input(shape=(original_dim,),name='input')
h = Dense(intermediate_dim,activation='relu',name='encoding')(x)
z_mean = Dense(latent_dim,name='mean')(h)
z_log_var = Dense(latent_dim, name='log-varience')(h)
z = Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_var])
encoder = Model(x, [z_mean,z_log_var,z], name="encoder")
{% endhighlight %}
<br>
디코더도 선언해준다.
<br>
{% highlight python %}
input_decoder = Input(shape = (latent_dim,),name='decoder_input')
decoder_h = Dense(intermediate_dim,activation='relu',name='decoder_h')(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid',name = "flat_decoded")(decoder_h)
decoder = Model(input_decoder,x_decoded, name = 'decoder')
{% endhighlight %}
<br>
이제 인코더와 디코더를 합쳐 VAE를 만들어준다
<br>
{% highlight python %}
output_combined = decoder(encoder(x)[2])
vae = Model(x,output_combined)
vae.summary()
{% endhighlight %}

<br>
사실 학습자체는 이미지를 넣어서 이미지랑 비교하는 모델이기 때문에 x_train만 있으면 충분하다. y_train과 y_test는 이번 코드에서는 사용하지 않는다.
<br>

{% highlight python %}
kl_loss = -0.5 * K.sum(1+z_log_var - K.exp(z_log_var)-K.square(z_mean),axis=-1)
vae.add_loss(K.mean(kl_loss)/784.)
vae.compile(optimizer='rmsprop', loss='binary_crossentropy')
{% endhighlight %}


{% highlight python %}
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(y_test), np.prod(x_test.shape[1:])))
vae.fit(x_train,x_train,shuffle=True,epochs=50,batch_size=batch_size)
{% endhighlight %}

학습하는 그래프를 보여주긴 하는데 1 epoch 이후에는 크게 변화가 없기도 했고, 이 변화하는 모습을 너무 찍고 싶어졌다. 그런데 pytorch는 매 epoch을 for로 처리하니까 간단하게 끼워넣을 수 있었는데 tensorflow에서는 fit을 쓰다보니 어떻게 넣어야할지 난감했다  
<br>
하지만 역시 구글링을 열심히 하면 답이 나온다!  
<br>
from tensorflow.keras.callbacks import Callback 을 선언해서 Callback 클래스를 상속 받으면 모델이 학습할 때, epoch이나 batch 숫자에 멤버변수로 접근할 수 있게된다!  
<br>
epoch은 너무 크게 변해버려서 나는 batch 마다 이미지를 저장하게 했다


{% highlight python %}
# Callback class 상속
class LatentSpaceVisualization(Callback):
    def __init__(self, x_data,y_data,cmap= 'viridis',color_bar_activation = False):
        self.x_data = x_data
        self.y_data = y_data
        self.color_bar_activation = color_bar_activation

    # batch 시작 때마다 테스트 데이터 분포 이미지 저장하기
    def on_train_batch_begin(self,batch,lgs=None):
        plt.figure(figsize=(10, 10))
        plt.title(f"batch : {batch:02d}")
        part_model = self.model.get_layer(name='encoder')
        y_pred=part_model(self.x_data)[0]
        plt.scatter(y_pred[:,0], y_pred[:,1], c=self.y_data, cmap='viridis')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        if self.color_bar_activation: plt.colorbar()
        plt.savefig(f'./vae_img/img_{batch:04d}.jpg')

# 클래스 선언
visualization=LatentSpaceVisualization(x_test, y_test)
# 학습시 콜백함수 지정
vae.fit(x_train,x_train,shuffle=True,epochs=1,batch_size=batch_size,callbacks=[visualization])

# 저장한 이미지 읽어서 gif 만들기
images = []
for i in range(0,600,3):
    img = cv2.imread(f'vae_img/img_{i:04d}.jpg')
    images.append(img)
    
imageio.mimsave('./result.gif',images,fps=15)
{% endhighlight %}
<br>
짜잔!

<br>
꿈틀거리는게 뭔가 귀엽다(?)
{% highlight python %}

{% endhighlight %}

{% highlight python %}

{% endhighlight %}

{% highlight python %}

{% endhighlight %}

![image](/public/img/vae_train.gif){: .center}

## 전체 코드 
- - - 
### VAE 학습 코드
{% highlight python %}

# 프레임워크 호출
from tensorflow.keras.layers import Input,Dense,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 하이퍼 파라미터 선언
batch_size = 100
original_dim =784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.

# 잠재변수 z를 위한 샘플링 함수 선언
def sampling(args:tuple):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim),mean=0.,stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 인코더 선언 부분
x = Input(shape=(original_dim,),name='input')
h = Dense(intermediate_dim,activation='relu',name='encoding')(x)
z_mean = Dense(latent_dim,name='mean')(h)
z_log_var = Dense(latent_dim, name='log-varience')(h)
z = Lambda(sampling,output_shape=(latent_dim,))([z_mean,z_log_var])
encoder = Model(x, [z_mean,z_log_var,z], name="encoder")

# 디코더 선언 부분
input_decoder = Input(shape = (latent_dim,),name='decoder_input')
decoder_h = Dense(intermediate_dim,activation='relu',name='decoder_h')(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid',name = "flat_decoded")(decoder_h)
decoder = Model(input_decoder,x_decoded, name = 'decoder')

# VAE = encoder + decoder
# VAE 모델 구성 부분
output_combined = decoder(encoder(x)[2])
vae = Model(x,output_combined)
vae.summary()

# 커스텀 loss 선언
kl_loss = -0.5 * K.sum(1+z_log_var - K.exp(z_log_var)-K.square(z_mean),axis=-1)
vae.add_loss(K.mean(kl_loss)/784.)
vae.compile(optimizer='rmsprop', loss='binary_crossentropy')

# MNIST 데이터 불러오기
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# np.prod = 배열의 원소끼리 곱해줌
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(y_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,x_train,shuffle=True,epochs=50,batch_size=batch_size)

{% endhighlight %}

### 학습 과정을 gif로 만들기 위한 Custom Callback class 만들기

{% highlight python %}

# Callback class 상속
class LatentSpaceVisualization(Callback):
    def __init__(self, x_data,y_data,cmap= 'viridis',color_bar_activation = False):
        self.x_data = x_data
        self.y_data = y_data
        self.color_bar_activation = color_bar_activation

    # batch 시작 때마다 테스트 데이터 분포 이미지 저장하기
    def on_train_batch_begin(self,batch,lgs=None):
        plt.figure(figsize=(10, 10))
        plt.title(f"batch : {batch:02d}")
        part_model = self.model.get_layer(name='encoder')
        y_pred=part_model(self.x_data)[0]
        plt.scatter(y_pred[:,0], y_pred[:,1], c=self.y_data, cmap='viridis')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        if self.color_bar_activation: plt.colorbar()
        plt.savefig(f'./vae_img/img_{batch:04d}.jpg')

# 클래스 선언
visualization=LatentSpaceVisualization(x_test, y_test)
# 학습시 콜백함수 지정
vae.fit(x_train,x_train,shuffle=True,epochs=1,batch_size=batch_size,callbacks=[visualization])

# 저장한 이미지 읽어서 gif 만들기
images = []
for i in range(0,600,3):
    img = cv2.imread(f'vae_img/img_{i:04d}.jpg')
    images.append(img)
    
imageio.mimsave('./result.gif',images,fps=15)

{% endhighlight %}