
# coding: utf-8

# - Reference : https://www.kaggle.com/jannesklaas/a-simple-seq2seq-translator/notebook
# -             https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# - 그림출처 : https://towardsdatascience.com/neural-machine-translation-using-seq2seq-with-keras-c23540453c74

# # Keras English to French translator 만들기
# - 우리가 구성할 모델은 seqeunce-2-sequence 모듈을 이용하여 영어 문장을 프랑스어 문장으로 바꿔주는 기계 번역기를 만드는 것입니다.
# - 본 실습에서는 기본적인 seq2seq 모듈과 attention 방식을 적용한 2가지 발전된 seq2seq모듈을 구현해보고자 합니다.

# In[4]:


# keras를 비롯하여 필요한 모듈들을 불러옵니다.
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
import os
from numpy.random import seed
seed(1)
import matplotlib.pyplot as plt
from tensorflow import set_random_seed
set_random_seed(2)


# In[ ]:


batch_size = 64   # Batch size for training.
epochs = 25       # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = './fra-eng/fra.txt'


# ### 데이터셋 예시
# - 아래 표는 model training에 사용될 문구들의 예시입니다.

# In[2]:


df = pd.read_csv(data_path,delimiter='\t')
df.head(20)


# In[3]:


del df # Save memory


# ### 데이터셋 가져오기 

# In[5]:


input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

lines = open(data_path, 'rt', encoding='UTF8').read().split('\n')
# 모든 pair 중 앞의 10000쌍 만 골라서 학습에 사용하도록 하겠습니다.
for line in lines[: min(num_samples, len(lines) - 1)]:
    # Input (English)과 target (French)을 분리합니다.
    input_text, target_text = line.split('\t')
    
    # Target text에서 "tab"을 "start sequence" character로, "\n" as "end sequence" character로 둡니다.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    
    # input, output 문장들을 구성하는 token들의 집합
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
            
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

print('Number of samples:', len(input_texts))


# ### 문자의 벡터화
# - 모델을 트레이닝 하기 위해서 입력된 문자들을 벡터 형태로 바꿔주는 것이 필요합니다.
# - 문자를 해당하는 벡터로 바꾸는 방식은 여러 가지 방법이 있지만, 여기서는 하나의 위치만 1을 나타내고 나머지는 0인 one-hot vector로 encoding하겠습니다.

# In[7]:


# Input과 target 문자들을 순서대로 정렬해줍니다.
input_characters = sorted(list(input_characters)) 
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)  # aka size of the english alphabet + numbers, signs, etc.
num_decoder_tokens = len(target_characters) # aka size of the french alphabet + numbers, signs, etc.

# one-hot encoding으로 생성한 vector의 dimension이 됩니다.
# 즉 여기서는 input (알파벳 및 문장부호)의 경우 71 dimension의 vector, output (프랑스어 알파벳 및 문장부호)은 93 vector dimension의 vector로 바꿀 것입니다.
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)


# In[8]:


# 각 token에 대응하는 숫자를 생성합니다.
input_token_index = {char: i for i, char in enumerate(input_characters)}
target_token_index = {char: i for i, char in enumerate(target_characters)}
for c in 'the cat sits on the mat':
    print(input_token_index[c], end = ' ')


# In[9]:


# 사용할 데이터 중 가장 긴 문구의 길이(글자수)를 가져옵니다.
max_encoder_seq_length = max([len(txt) for txt in input_texts]) 
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[10]:


# encoder_input_data는 (num_pairs, max_english_sentence_length, num_english_characters) 형태의 3D array
# 즉 영어문장의 one-hot vector들로 구성됩니다.

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

# decoder_input_data 역시 (num_pairs, max_french_sentence_length, num_french_characters) 형태의 3D array
# 프랑스어 문장의 one-hot vector들로 구성됩니다.

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# decoder_target_data는 input과 형태는 같지만 input에서 한 time step 뒤의 값을 가집니다.
# 예를 들어 'abc'라는 decoder input sequence가 있다면, target data는 기존 값에서 모델을 통해 예측된 문자 'd'를 덧붙인 'abcd'가 될 것입니다.

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


# In[11]:


# 문장 내 각 token에 대응하는 One-hot 벡터를 만드는 과정입니다.
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data는 
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# ### 모델 구성하기
# 
# - 아래 코드에 training을 위한 모델을 소개하였습니다.
# - 성능을 높이기 위해 decoder 단에서 입력된 target 값에서 한 time step 씩 뒤의 값들을 decoder target 값으로 사용합니다. 즉 decoder input/output에서 모두 labeled 된 값을 사용합니다.
# ![seq2seq_training](./seq2seq_training.png)

# In[12]:


def seq2seq_train(num_encoder_tokens, num_decoder_tokens, latent_dim):
    # Encoder
    # return_state argument를 True로 설정하여 LSTN layer의 마지막 time step의 내부 state들을 얻습니다.
    encoder_inputs = Input(shape=(None, num_encoder_tokens), 
                           name = 'encoder_inputs')
    encoder = LSTM(latent_dim, 
                        return_state=True, 
                        name = 'encoder')

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder 
    # encoder_states를 decoder 단의 lstm에 초기 state로 이용합니다.
    decoder_inputs = Input(shape=(None, num_decoder_tokens), 
                           name = 'decoder_inputs')

    decoder_lstm = LSTM(latent_dim, 
                             return_sequences=True, 
                             return_state=True, 
                             name = 'decoder_lstm')

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, 
                          activation='softmax', 
                          name = 'decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense, model
[encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense, model] = seq2seq_train(num_encoder_tokens, num_decoder_tokens, latent_dim)


# In[13]:


# 저장된 모델을 불러오는 부분입니다.
project_path = './model/'

if not os.path.exists(project_path):
    os.mkdir(project_path)
    
model_path = project_path + 's2s.hdf5'

checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


# In[14]:


# 모델 학습하기 (시간상 넘어가도록 하겠습니다.)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
history = model.fit([encoder_input_data, decoder_input_data], 
                    decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    callbacks=[checkpoint])
# 학습한 모델 저장하기
model.save(model_path)


# In[14]:


# Training curve를 그리는 코드입니다. 모델이 트레이닝 될 경우에만 그려주므로 넘어가도록 합니다.
plt.figure(figsize=(10,7))
a, = plt.plot(history.history['loss'],label='Training Loss')
b, = plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend(handles=[a,b])
plt.show()


# ![Curve](./seq2seq_training_curve.png)

# ## Inference (sampling) model 만들기
# - 실제 우리가 사용할 모델은 decoder단에서 예측한 문자가 다시 lstm의 input으로 들어가 다음 글자를 예측하는 과정이 반복되는 방식을 사용합니다. 
# ![seq2seq_inference](./seq2seq_inference.png)

# In[15]:


model = load_model(model_path)
def seq2seq_inference(latent_dim, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense):
    '''Encoder model'''
    encoder_model = Model(encoder_inputs, encoder_states)

    '''Decoder model'''
    # Decoder의 input
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Decoder output 및 state
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # 다음 문자 예측
    decoder_outputs = decoder_dense(decoder_outputs)

    # Decoder 모델
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return [encoder_model, decoder_model]
[encoder_model, decoder_model] = seq2seq_inference(latent_dim, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense)
# [encoder_model, decoder_model] = seq2seq(num_encoder_tokens, num_decoder_tokens, latent_dim).model_inference(encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense)


# In[16]:


# decode된 각 숫자에 대응하는 token입니다.
reverse_input_char_index = {i: char 
                            for char, i in input_token_index.items()}
reverse_target_char_index = {i: char 
                             for char, i in target_token_index.items()}
print(reverse_input_char_index)
print(reverse_target_char_index)


# ### 문자 decoding
# 이제 학습한 모델을 이용해 영어문장을 프랑스어 문장으로 바꿔봅시다.

# In[17]:


def decode_sequence(input_seq):
    # Input을 state vector들로 encode 합니다.
    states_value = encoder_model.predict(input_seq)

    # Decoder input sequence의 첫 부분을 start character로 초기화합니다.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    
    # stop sign이 나오기 전까지 문장을 예측합니다.
    while not stop_condition:
        # Decoder의 이전 output, internal states를 이용해 다음 값을 예측합니다.
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 가장 높은 확률로 예측되는 token을 sampling합니다.
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # 예측한 token에 해당하는 문자를 찾습니다.
        sampled_char = reverse_target_char_index[sampled_token_index]
        # decoded된 문장에 예측 문자를 덧붙여 문장을 예측해갑니다.
        decoded_sentence += sampled_char

        # decoded된 문장이 최대 길이를 초과하거나 'stop' character를 만나게 될 때 예측을 멈춥니다.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Target sequence를 update합니다.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 다음 문자를 예측하기 위해 state 값을 update합니다.
        states_value = [h, c]

    return decoded_sentence


# In[18]:


my_text = 'Run!'
placeholder = np.zeros((1,len(my_text)+10,num_encoder_tokens))


# In[19]:


for i, char in enumerate(my_text):
    print(i,char, input_token_index[char])
    placeholder[0,i,input_token_index[char]] = 1


# In[20]:


decode_sequence(placeholder)


# In[22]:


for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

