# Deep-Learning
Deep learning (also known as deep structured learning) is part of a broader  family of machine learning methods based on artificial neural networks with  representation learning. Learning can be supervised, semi-supervised or unsupervised.  Deep-learning architectures such as deep neural networks, deep belief  networks, recurrent neural networks and convolutional neural networks  have been applied to fields including computer vision, machine vision,  speech recognition, natural language processing, audio recognition,  social network filtering, machine translation, bioinformatics, drug  design, medical image analysis, material inspection and board game  programs, where they have produced results comparable to and in some  cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing  and distributed communication nodes in biological systems. ANNs have  various differences from biological brains. Specifically, neural networks  tend to be static and symbolic, while the biological brain of most living  organisms is dynamic (plastic) and analogue.

The project is created with Anaconda. Anaconda is a distribution of the Python and R programming languages for scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, etc.), that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS. It is developed and maintained by Anaconda, Inc., which was founded by Peter Wang and Travis Oliphant in 2012. As an Anaconda, Inc. product, it is also known as Anaconda Distribution or Anaconda Individual Edition, while other products from the company are Anaconda Team Edition and Anaconda Enterprise Edition, both of which are not free. 
To run this Project you must download Anaconda latest version from his website for Windows, Mac or Linux. You have uninstall any python form your computer and install clean Anaconda.

Aster you install Anaconda search Anaconda Navigator from taskbar and run it. After its open there is menu --> go to Enviroments -->and then create new envirement, name it Tensorflow or whathever you want. Wait a minute and then from the drop menu change from installed to not insatalled and search for Tensorflow , and chek and apply, wait a few minutes the do the same thing, searching for Keras. After raady with that go back and you will see Jupyter notebook is not insatlled, beacause you've create new envirement, click on install
Now what exacly is my project? The project creates similar text to text i give him to him, i choose Shakespeare Texts. But it can work with any texts without problem.
first you have to import these Lbraries:
  
```python
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
```
For IBM you have to <b>!pip install ibm_watson</b>

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications. And Thats why he is essential for thsi project
After imports are made now we can get a text file and i choose this : https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

```python
filepath = tf.keras.utils.get_file('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8')
```
The problem that we have right now with our data is that we are dealing with text. We cannot just train a neural network on letters or sentences. We need to convert all of these values into numerical data. So we have to come up with a system that allows us to convert the text into numbers, then predict specific numbers based on that data and then again convert the resulting numbers back into text.

```python
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[200000:900000]
```
Here we select all the characters from character number 200,000 up until 900,000. So we are processing a total of 700,000 characters, which should be enough for pretty descent results.

```python
characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))
```
Now we create a sorted set of all the unique characters that occur in the text. In a set no value appears more than once, so this is a good way to filter out the characters. After that we define two structures for converting the values. Both are dictionaries that enumerate the characters. In the first one, the characters are the keys and the indices are the values. In the second one it is the other way around. Now we can easily convert a character into a unique numerical representation and vice versa.

```python
x = np.zeros((len(sentences), SEQ_LENGTH,
              len(characters)), dtype=np.bool)
y = np.zeros((len(sentences),
              len(characters)), dtype=np.bool)

for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1
```
This might seem a little bit complicated right now but it is not. We are creating two NumPy arrays full of zeros. The data type of those is bool, which stands for boolean. Wherever a character appears in a certain sentence at a certain position we will set it to a one or a True. We have one dimension for the sentences, one dimension for the positions of the characters within the sentences and one dimension to specify which character is at this position.
```python
model = Sequential()
model.add(LSTM(128,
               input_shape=(SEQ_LENGTH,
                            len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
```
The inputs immediately flow into our LSTM layer with 128 neurons. Our input shape is the length of a sentence times the amount of characters. The character which shall follow will be set to True or one. This layer is followed by a Dense hidden layer, which just increases complexity. In the end we use the Softmax activation function in order to make our results add up to one. This gives us the probability for each character.

```python
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```
It basically just picks one of the characters from the output. As parameters it takes the result of the prediction and a temperature. This temperature indicates how risky the pick shall be. If we have a high temperature, we will pick one of the less likely characters. A low temperature will cause a conservative choice.

```python
def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions,
                                 temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated
```

So basically the first SEQ_LENGTH amount of characters will be copied from the original text. But we could just cut them off afterwards and we would end up with text that is completely generated by our neural network.

So we choose some random starting text and then we run a for loop in the range of the length that we want. We can generate a text with 100 characters or one with 20,000. We then convert our sentence into the desired input format that we already talked about. The sentence is now an array with ones or Trues, wherever a character occurs. Then we use the predict method of our model, to predict the likelihoods of the next characters. Then we make use of our sample helper function. In this function we also have a temperature parameter, which we can pass to that helper function. Of course the result we get needs to be converted from the numerical format into a readable character. Once this is done, we add the character to our generated text and repeat the process, until we reach the desired length.

Now but not least, the generated text is saved to  .mp3 file thanks to IBM cloud services. And the service we use is Text-to speech

```python
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
```

it basically returns a clear speech from the text that is created.
