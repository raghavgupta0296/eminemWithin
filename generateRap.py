import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import pronouncing as pr
import random

look_back = 100

f = open("cleanedRapLyrics.txt", 'r', encoding='utf-8')
lyrics = f.read()
f.close()
dictionary = list(set(lyrics.split()))
lyrics = lyrics[::-1]
chars = sorted(set(lyrics))
char2int = dict(zip(chars, range(len(chars))))
int2char = dict(zip(char2int.values(), char2int.keys()))

data = [char2int[i] for i in lyrics]

model = Sequential()
model.add(LSTM(512, input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(BatchNormalization())
model.add(Activation('softmax'))

filepath = "weights-improvement-21-1.5115-bigger.hdf5"
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer='adam')

rhymeSwitch = True
def rhymeWord(word_to_rhyme):
    global rhymeSwitch
    if rhymeSwitch == True:
        words = pr.rhymes(word_to_rhyme)
        if len(words)!=0:
            r = random.choice(words)
            ctr=0
            while r not in dictionary:
                r = random.choice(words)
                ctr+=1
                if ctr>30:
                    r = random.choice(words)
                    break

        else:
            r = random.choice(dictionary)
    else:
        r = random.choice(dictionary)
    return r

start_seed = np.random.randint(0, len(data) - look_back)
pattern = data[start_seed:start_seed + look_back]
y = [int2char[i] for i in pattern]
y = "".join(y) + ""
print(y[::-1])
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    pred = model.predict(x,verbose=0)
    p = np.argmax(pred)
    pattern.append(p)
    pattern = pattern[1:]
    y = y + int2char[p]
    # if int2char[p] == '\n':
    #     sentences = y[::-1].split("\n")
    #     word_to_rhyme = sentences[1].split(" ")[-1]
    #     if word_to_rhyme!=",":
    #         rhyming_word = rhymeWord(word_to_rhyme)
    #     else:
    #         word_to_rhyme = sentences[1].split(" ")[-2]
    #         rhyming_word = rhymeWord(word_to_rhyme)
    #     print("Word to rhyme : ",word_to_rhyme)
    #     print("rhyming word : ",rhyming_word)
    #     if i%3==0 and rhymeSwitch==True:
    #         rhymeSwitch = not rhymeSwitch
    #     else:
    #         rhymeSwitch = True
    #     for rw in rhyming_word[::-1]:
    #         pattern.append(char2int[rw])
    #         y = y + rw
    #     y = y + " "
    #     pattern.append(char2int[" "])
    #     pattern = pattern[len(rhyming_word)+1:]
print(y[::-1])
