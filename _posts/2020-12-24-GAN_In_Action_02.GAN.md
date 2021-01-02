---
layout: post
title:  "02.GAN"
date:   2020-12-24 23:42:29 +0900
categories: GAN_In_Action
---

# GAN
---
GAN은 생성자(Generator)와 판별자(Discriminator) 총 2가지 모델로 구성된다. 생성자가 만든 결과물을 판별자가 판별하면서 학습하게 되며, 원 논문저자인 이안 굿펠로는 위조지폐범(생성자)와 경찰(판별자)를 비유로 들며, 위조 지폐범과 경찰이 서로 경쟁하면서 학습을 하게되면, 내쉬균형(Nash equilibrium) 상태가 
되는(판별자의 acc가 50%가 되는) 균형점을 찾는
<br>
생성자와 판별자는 각각의 loss 함수를 가지고 있으며, 판별자의 loss를 사용해 네트워크 전체를 훈련하게 된다.
<br>
 ![image](/public/img/gan.png){: width="100%" height="100%"}{: .center}

<br>

# GAN 훈련 

1. 판별자 훈련
    - 진짜 이미지의 미니배치 x를 받기
    - 랜덤 노이즈 z의 미니배치를 받고 가짜 샘플의 미니배치를 생성 G(z) = x*
    -  D(x)와 D(x*)에 대한 분류 손시을 계싼하고 전체 오차를 역전파하여 분류 손실을 최소화되도록 업데이트
<br>
2. 생성자 훈련
    - 랜덤 노이즈 z의 미니배치를 받고 가짜 샘플의 미니배치 생성 G(z) = x*
    - D(x*)에 대한 분류 손실을 계산하고 오차를 역전파하여 손실을 최대화 하도록 업데이트
 
{% highlight python %}
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)
z_dim = 100

def build_generator(img_shape,z_dim):
    model = Sequential()
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(np.prod(img_shape),activation='tanh'))
    model.add(Reshape(img_shape))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape = img_shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1,activation='sigmoid'))
    return model

def build_gan(generator,discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])
generator = build_generator(img_shape,z_dim)
discriminator.trainable = False

gan = build_gan(generator,discriminator)
gan.compile(loss='binary_crossentropy',optimizer=Adam())

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
iteration_checkpoints = []

def train(iterations,batch_size,sample_interval):
    (x_train,_),(_,_) = mnist.load_data()
    x_train = x_train / 127.5-1.0
    x_train = np.expand_dims(x_train, axis=3)
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    
    for iteration in range(iterations):
        idx = np.random.randint(0,x_train.shape[0],batch_size)
        imgs = x_train[idx]
        
        z= np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real,d_loss_fake)
        
        
        g_loss = gan.train_on_batch(z,real)
        
        if (iteration+1) % sample_interval ==0:
            losses.append((d_loss,g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)
            
            print(f"{iteration+1} [D 손실: {d_loss:f}, 정확도: {100.0*accuracy}][G 손실: {g_loss}]")
            sample_images(generator)

iterations = 40000
batch_size = 128
sample_interval = 1000
train(iterations, batch_size, sample_interval)

 {% endhighlight %}

 ![image](/public/img/gan_result.png){: width="100% height="100%"}{: .center}

 생각보다 결과가 잘 안나온다 ㅠㅠ  
 GAN은 다른 네트워크에 비해 학습 자체가 좀 어렵고 이를 대비하기 위해 여러 방법들이 나오고 있다고 한다.  
 언젠간 제대로 된 이미지를 생성할 수 있기를 바래본다 ㅠㅠ  
