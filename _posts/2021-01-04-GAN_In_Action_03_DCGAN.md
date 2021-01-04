---
layout: post
title:  "03.DCGAN"
date:   2020-12-24 23:42:29 +0900
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

{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten,LeakyReLU, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows,img_cols,channels)
z_dim =100

def build_generator(z_dim):
    model = Sequential()
    model.add(Dense(256*7*7, input_dim=z_dim))
    model.add(Reshape((7,7,256)))
    
    model.add(Conv2DTranspose(128,kernel_size=3, strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(64, kernel_size=3,strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(1,kernel_size=3,strides=2,padding='same'))
    model.add(Activation('tanh'))
    
    return model

def build_discriminator(img_shape):
    model = Sequential()
    
    model.add(Conv2D(32,kernel_size=(3,3),strides=2,input_shape=img_shape,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(64,kernel_size=(3,3),strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2D(128,kernel_size=(3,3),strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

generator = build_generator(z_dim)
discriminator.trainable=False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer = Adam())

def sample_images(generator, image_grid_rows=4,image_grid_columns=4):
    z = np.random.normal(0,1,(image_grid_rows * image_grid_columns,z_dim))
    
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig,axs = plt.subplots(image_grid_rows,image_grid_columns,figsize=(4,4),sharey=True,sharex=True)
    
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt+=1

losses = []
accuracies = []
iteration_check_points = []

def train(iterations, batch_size, sample_interval):
    (x_train,_),(_,_) = mnist.load_data()
    
    x_train = x_train / 127.5 - 1.0
    x_train = np.expand_dims(x_train,axis=3)
    
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    
    for iteration in range(iterations):
        idx = np.random.randint(0,x_train.shape[0],batch_size)
        imgs = x_train[idx]
        
        z=np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(z)
        
        d_loss_real = discriminator.train_on_batch(imgs,real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs,fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real,d_loss_fake)
        
        g_loss = gan.train_on_batch(z,real)
        
        if(iteration+1) % sample_interval == 0:
            losses.append((d_loss,g_loss))
            accuracies.append(100.0*accuracy)
            iteration_check_points.append(iteration+1)
            print(f"{iteration+1} [D 손실: {d_loss:f}, 정확도: {100.0*accuracy}][G 손실: {g_loss}]")
            sample_images(generator)

iterations = 20000
batch_size = 128
sample_interval = 1000


train(iterations,batch_size,sample_interval)

{% endhighlight %}

![image](/public/img/dcgan_result.png){: width="100%" height="100%"}{: .center}
<br>
확실히 바닐라 gan보다는 이미지 품질이 좋게 나오는거 같다.  
gan으로 생성한 이미지는 일단 학습 자체가 어려워서 성공을 못했었는데, 다행히 dcgan은 이미지 생성에 성공해서, mnist에 실제로 있는 데이터 처럼 보인다.  
나중엔 더 큰 이미지도 성공해 보고 싶다!  