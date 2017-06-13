import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization


f = open("cleanedRapLyrics.txt",'r',encoding='utf-8')
lyrics = f.read()
f.close()
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
char2int = dict(zip(chars,range(len(chars))))
int2char = dict(zip(char2int.values(),char2int.keys()))

data = [char2int[i] for i in lyrics]

def one_hot(i):
    hotVec = np.zeros(shape=(len(chars)))
    hotVec[i] = 1
    return hotVec

look_back = 100

X = []
Y = []
for i in range(len(data)-look_back):
    x=data[i:i+look_back]
    X.append(x)
    Y.append(one_hot(data[i+look_back]))

X = np.array(X)
Y = np.array(Y)
X = np.reshape(X,(X.shape[0],X.shape[1],1))
X = X/float(len(chars))
print("X shape : ",X.shape)

model = Sequential()
model.add(LSTM(512,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(BatchNormalization())
model.add(Activation('softmax'))
filepath = "weights-improvement-37-1.2212-bigger.hdf5"
model.load_weights(filepath)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

filepath="/output/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, Y, nb_epoch=50, batch_size=1000, callbacks=callbacks_list)
