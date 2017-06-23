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
lines = lyrics.split("\n")
lyrics = ""
for l in lines:
    try:
        if l[-1] == " ":
            l = l[:-1]
    except:
        pass
    lyrics = lyrics+l[::-1]+"\n"
lyrics = lyrics[:-1]
chars = sorted(set(lyrics))
char2int = dict(zip(chars, range(len(chars))))
int2char = dict(zip(char2int.values(), char2int.keys()))

data = [char2int[i] for i in lyrics]


# y = input("Enter lyrics: ")
# lines2 = y.split("\n")
# y = ""
# for l in lines2:
#     try:
#         if l[-1] == " ":
#             l = l[:-1]
#     except:
#         pass
#     y = y+l[::-1]+"\n"
# y = y[:-1]
# print (y)
# pattern = []
# for p in y:
#     pattern.append(char2int[p])
# look_back = len(y)
# print("lookback : ",look_back)


model = Sequential()
model.add(LSTM(512, input_shape=(look_back, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(BatchNormalization())
model.add(Activation('softmax'))

filepath = "weights-improvement-31-1.2793-bigger.hdf5"
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

rhymeSwitch = True
usedRhymeWords = []
def rhymeWord(word_to_rhyme):
    global rhymeSwitch
    global usedRhymeWords
    if rhymeSwitch == True:
        words = pr.rhymes(word_to_rhyme)
        if len(words)!=0:
            # r = random.choice(words)
            candidates = []
            for w1 in words:
                if w1 in dictionary and w1 not in usedRhymeWords and len(w1)!=1:
                    candidates.append(w1)
            print("candidates : ",candidates)
            if len(candidates)==0:
                r=0
            else:
                r = np.random.choice(candidates)
                usedRhymeWords.append(r)
        else:
            r = 0
    else:
        r = 0
    return r

start_seed = np.random.randint(0, len(data) - look_back)
pattern = data[start_seed:start_seed + look_back]
y = [int2char[i] for i in pattern]
y = "".join(y) + ""
print(y[::-1])
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    if i==0:
        pred = "\n"
    else:
        pred = model.predict(x,verbose=0)
    p = np.argmax(pred)
    pattern.append(p)
    pattern = pattern[1:]
    y = y + int2char[p]
    if int2char[p] == '\n':
        # sentences = y[::-1].split("\n")
        sentence = y.split("\n")[-2]
        word_to_rhyme = sentence[::-1].split(" ")[-1]
        # if word_to_rhyme!=",":
        rhyming_word = rhymeWord(word_to_rhyme)
        # else:
        #     word_to_rhyme = sentence[1].split(" ")[-2]
        #     rhyming_word = rhymeWord(word_to_rhyme)
        print("Word to rhyme : ",word_to_rhyme)
        print("rhyming word : ",rhyming_word)
        if i%3==0 and rhymeSwitch==True:
            rhymeSwitch = not rhymeSwitch
        else:
            rhymeSwitch = True
        if rhyming_word == 0:
            continue
        for rw in rhyming_word[::-1]:
            pattern.append(char2int[rw])
            y = y + rw
        y = y + " "
        pattern.append(char2int[" "])
        pattern = pattern[len(rhyming_word)+1:]

        if len(rhyming_word)<=3:
            rhymeSwitch = True
            word_to_rhyme = sentence[::-1].split(" ")[-2]
            rhyming_word = rhymeWord(word_to_rhyme)
            print("Word to rhyme 2: ", word_to_rhyme)
            print("rhyming word 2: ", rhyming_word)
            if rhyming_word == 0:
                continue
            for rw in rhyming_word[::-1]:
                pattern.append(char2int[rw])
                y = y + rw
            y = y + " "
            pattern.append(char2int[" "])
            pattern = pattern[len(rhyming_word) + 1:]

y = y.split("\n")
for i in y:
    print(i[::-1])

