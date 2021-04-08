#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


filepath = tf.keras.utils.get_file('shakespere.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()




text = text[300000:800000]
characters = sorted(set(text))

char_to_index = dict((c,i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i, c in enumerate(characters))

SEQ_LENGHT = 40
STEP_SIZE = 3

sentences = []
next_character = []

'''
for i in range(0, len(text) - SEQ_LENGHT, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGHT])
    next_character.append(text[i+SEQ_LENGHT])


x = np.zeros((len(sentences), SEQ_LENGHT, len(characters)), dtype = np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentance in enumerate(sentences):
    for t, character in enumerate(sentance):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_character[i]]] = 1




model = Sequential()
model.add(LSTM(128, input_shape = (SEQ_LENGHT, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))


model.compile(loss = 'categorical_crossentropy' , optimizer=RMSprop(lr = 0.01))

model.fit(x, y, batch_size = 256 , epochs = 4)

model.save('textgenerator.model')

'''

model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGHT - 1)
    generated = ''
    sentance = text[start_index: start_index + SEQ_LENGHT]
    generated += sentance
    for i in range(length):
        x = np.zeros((1, SEQ_LENGHT, len(characters)))
        for t, character in enumerate(sentance):
            x[0, t, char_to_index[character]] = 1
        
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]
        
        generated += next_character
        sentance = sentance[1:] + next_character
    return generated




myTxt = generate_text(300, 0.60)
print(myTxt)

url = 'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/ba6d95ea-0c41-4aa9-a880-d5a4852a6181'
apikey = 'MFo4bFpSBQujBs8F_q0fEoVDluV6oMzG8lqVs5Uu186H'
authenticator = IAMAuthenticator(apikey)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)
with open('./shakespire.mp3', 'wb') as audio_file:
    res = tts.synthesize(myTxt, accept='audio/mp3', voice='en-GB_JamesV3Voice').get_result()
    audio_file.write(res.content)




    


# In[ ]:





# In[ ]:




