'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.

https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

#path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
filename = "nietzsche"
text = open(filename+'.txt',encoding='utf-8').read().lower().replace("\n",' ').replace("--"," ").replace("  "," ").replace('\r\n',' ')
texts = text.split(' ')
print("texts length:",len(texts))
#词组超长，可能造成MemeryError,这里选择长度
text = texts[:4000]
print(texts[:100])
print("100 chars=>",text[:100])
print('corpus length:', len(text))

#使用字符数量
chars = sorted(list(set(text)))
print('total chars:', len(chars))

#print("chars=>",chars)


char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print('char_indices length:',len(char_indices),"char_indices type",type(char_indices))
print('indices_char length:',len(indices_char))

#print(indices_char)
#print(char_indices)

#exit()
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
leng = 5000
dlen = 3800
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

#print(next_chars)
print('nb sequences:', len(sentences))



print('Vectorization...')

print(len(sentences), maxlen, len(chars))



X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# X = np.zeros((leng, maxlen, dlen), dtype=np.bool)
# y = np.zeros((leng, dlen), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.save("ssby.model")


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
filenameid = 1
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)


    for diversity in [2.4]:
        print()
        print('----- diversity:', diversity,'--------------')

        generated = ' '
        sentence = text[start_index: start_index + maxlen] #变换后的text为数组，下面用join拼接成
        #generated += sentence
        temp_sentence = ' '.join(sentence)
        # temp_sentence = "Websites built using ASP.Net technologies are typically a nightmare for web scraping developers, mostly due to the way they handle forms."

        #generated += temp_sentence + ' '
        generated += temp_sentence  #中文不要空格
        print('----- Generating with seed: "' + temp_sentence + '--------------')
        sys.stdout.write(generated)


        
        w_chars = ""
        for i in range(600):
            x = np.zeros((1, maxlen, len(chars)))
            #for t, char in enumerate(sentence):
            list_sentence = temp_sentence.split(' ')  #重新构建list
            for t, char in enumerate(list_sentence):
                #print(char,t,type(char))
                #exit()

                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index] + ' '
            #print("next_char:",next_char)
            generated += next_char
            #sentence = sentence[1:] + next_char
            sentence = temp_sentence + next_char  + " "
            w_chars = w_chars + next_char + " "
            sys.stdout.write(next_char)
            sys.stdout.flush()
        fo =  open(filename+"-"+str(filenameid)+".txt","w",encoding='utf-8')
        fo.writelines(w_chars)

        fo.close()
        filenameid = filenameid +  1
        print()
