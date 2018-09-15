
# coding: utf-8

# # Keras를 이용한 RNN 구현
# 
# ### 이 실습의 목적
# - Keras 프레임워크를 이용하여, RNN을 구현하고, 이를 이용하여 imdb 데이터 셋을 이용하여 영화의 리뷰를 통해 평점을 예측하는 작업을 수행해봅니다.
# - 이 실습을 통해 RNN의 기본적인 원리들을 간단하게 복습하고, Keras에 대한 기초적인 경험을 쌓을 수 있습니다.

# ### RNN이란
# - Recurrent Neural Network의 약자로, 피드백 루프를 이용하여 순차적으로 데이터를 처리할 수 있는 네트워크입니다.
# - 일반적인 구조의 feedforward network나 CNN과는 달리 variable-length를 가지는 데이터들을 이용해 학습할 수 있습니다.
# - 일반적으로 크기가 제각기 다르고 순차적인 순서가 의미를 갖는 오디오나 언어처리 등의 영역에서 많이 사용됩니다.

# ### imdb는
# - 입력되는 input data는 영화에 대한 리뷰 문장입니다. 문장 텍스트 자체가 주어지지는 않고, 단어 인덱스의 sequence로 주어지게 됩니다.
#     - 예를 들어, 'It is really good movie'라는 문장을 단어 별로 쪼갠 후, [3, 1, 179, 32, 24]로 바꾸어 처리하게 됩니다.
# - 타겟으로 이용되는 target data는 해당 리뷰가 긍정적인지 부정적인지에 대한 라벨입니다. 긍정적이라면 1, 그렇지 않다면 0으로 되어 있습니다.

# ### 필요한 keras 모듈 / 패키지 가져오기

# In[1]:


import numpy as np
import keras
from keras import layers, models, datasets
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing import sequence

import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler

import os


# ### 데이터 셋 불러오기 및 패딩
# - 데이터를 keras 패키지의 함수를 이용해 불러온 뒤, 제일 긴 길이에 맞추어 0으로 패딩해줍니다.
# - num_words는 빈도 수 기준으로 몇 개의 단어를 사용할지를, max_len은 패딩의 기준을 의미합니다.

# In[2]:


max_len = 80

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=20000)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)


# ### 모델 구성하기
# - LSTM이라고 하는, 더 먼 시점의 데이터를 필요에 따라 더 잘 기억할 수 있는 유닛을 사용합니다.
# 
# - adam optimizer는 learning rate를 조절하여 학습이 잘 진행되게 해주는 조절 기법 중 하나입니다.

# In[3]:


x = layers.Input((max_len,))
h = layers.Embedding(20000, 128)(x)
h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
y = layers.Dense(1, activation='sigmoid')(h)

model = models.Model(x, y)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### 학습 진행하기
# - 학습이 진행되며 정확도가 함께 출력되게 됩니다.
# - batch_size는 한번에 몇 개의 샘플을 돌릴지를, epochs는 몇번의 iteration을 수행할지를 의미합니다.
# - verbose는 얼마나 자주 상태를 출력할지에 대한 파라미터입니다.

# In[ ]:


model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test), verbose=2)


# ### 정확도 살펴보기
# 학습데이터가 아닌 테스트데이터에 대한 정확도를 측정해봅시다.

# In[9]:


loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=2)


# In[10]:


loss, acc

