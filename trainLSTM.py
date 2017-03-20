from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from keras import backend as KB
from keras.layers.core import K
from keras.objectives import categorical_crossentropy
import tensorflow as tf
import numpy as np
import word2vec2

tf.set_random_seed(1001)

f = open("cleanedRapLyrics.txt")
data = f.read().lower()

d = []

try:
    data = data.split("\r\n")
    for i in data:
        for j in i.split(" "):
            d.append(j)
        d.append("\n")
except:
    print("error : modify code for spliting of data into words")

print("DATA TOKENIZED..............")

data = d
del d

# data = tf.compat.as_str((data.split()))

vocab_size = 5000
embedding_size = 128

data2, most_common, word2int, int2word = word2vec2.integerizeData(data, vocab_size)

del data

print("DATA INTEGERIZED............")

batch_size = 128
num_skips = 2
skip_window = 1

validation_set = np.random.choice(50, 3, replace=False)
neg_samples = 64

# embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embeddings = word2vec2.trainWord2Vec(data2, batch_size, validation_set, vocab_size, embedding_size, neg_samples,
                                     num_skips, skip_window, int2word)
print(embeddings)

print("Embeddings Obtained.........")

print(" Embeddings Shape : ", embeddings.shape)

# saver = tf.train.Saver()
# with tf.Session() as sess:
#	saver.restore(sess,"/tmp/word2vecModel.ckpt")
#	sess.run(tf.all_variables())
#	e = tf.get_collection(tf.GraphKeys.VARIABLES, "e")[0]
#	print "embeddings : ",sess.run(e)


words_to_read = 20

X = tf.placeholder(tf.float32, shape=(None, words_to_read, embedding_size))
Y = tf.placeholder(tf.float32, shape=(None,vocab_size))

def one_hot(num):
    a = np.zeros(vocab_size)
    a[num] = 1
    return a

i = 0

def newXY(i):
    # for i in range(0,len(data2)-words_to_read,1):
    # global i
    frame = data2[i]
    # X = tf.placeholder(tf.float32,shape=(words_to_read,embedding_size))
    # X = tf.placeholder(tf.float32,shape=(embedding_size,))
    # x = tf.nn.embedding_lookup(embeddings,frame)
    x = []
    # x = embeddings[frame]
    # initialize X to 0?
    frame = data2[(i):(i + words_to_read)]
    output = data2[i + words_to_read]
    # Y = tf.nn.embedding_lookup(embeddings,output)
    y = one_hot(output) #y = embeddings[output]
    for f in frame:
        # t = tf.nn.embedding_lookup(embeddings,f)
        x = np.append(x, embeddings[f])
    # print x.shape
    # print t,t.shape
    # x = tf.concat([x,t],0)
    # print "X shape : ",X.get_shape()
    # i+=1
    x = np.reshape(x, (words_to_read, embedding_size))
    # print x.shape
    return x, y


# nX = len(X)

# print "DATA VECTORIZED............."

# X = np.reshape(X,(nX,words_to_read,1))
# print X.get_shape(),Y.get_shape()

sess = tf.Session()
K.set_session(sess)

model = Sequential()
model.add(LSTM(256, input_shape=(int(X.get_shape()[1]), int(X.get_shape()[2])), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_size)) #, activation="softmax"))
# model.compile(loss="categorical_crossentropy",optimizer="adam")

# model = LSTM(256,input_shape=(X.get_shape()[1],X.get_shape()[2]),init='uniform',return_sequences=True)(X)
# model = Dropout(0.2)(model)
# model = LSTM(256)(model)
# model = Dropout(0.2)(model)
# model = Dense(embedding_size,activation='softmax')(model)

outModel = model(X)

# loss = tf.reduce_mean(categorical_crossentropy(Y, outModel))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outModel,Y))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

print("MODEL CREATED..............")

model_json = model.to_json()
with open("rapModel.json", "w") as json_file:
    json_file.write(model_json)

# f2 = "weights-improvement.hdf5"
# checkpoint = ModelCheckpoint(f2,monitor='loss',verbose=1,save_best_only=True,mode='min')

print("Training...................")


# model.fit(X,Y,nb_epoch=20,batch_size=128,callbacks=[checkpoint])
'''
def closestEmbedding(predictedWord):
    ctrEm=[]
    for q in range(len(embeddings)):
        ctr = 0
        for w in range(len(embeddings[q])):
            ctr += (predictedWord[0][w] - embeddings[q][w])**2
        ctrEm.append(ctr)
    predictedWord = np.argmax(ctrEm)
    # to not get UNK as prediction
    if (predictedWord == 0):
        predictedWord = np.argsort(ctrEm)[-2]
    #
    return predictedWord
'''
def generateTest():
    words_to_generate = 30
    testSeed = np.random.randint(low=1,high=(len(data2)-words_to_read),size=1)
    testSeed = int(testSeed)
    tx = []
    for t in range(testSeed,testSeed+words_to_read,1):
        tx.append(int2word[data2[t]])
    actualY = data2[testSeed+words_to_read]
    actualY = one_hot(int(actualY))
    actualY = np.reshape(actualY,(1,vocab_size))
    # tx = ["came", "back", "from", "those", "strange", "streets","\n","any","day","of","those","lame","dreams","\n","move","with","the","same","speed","\n"]
    print(tx)
    for i in range(len(tx)):
        try:
            tx[i] = word2int[tx[i]]
        except:
            tx[i] = word2int['UNK']
    # tx = [word2int[tx_i] for tx_i in tx]
    tx = [embeddings[tx_i] for tx_i in tx]
    for j in range(words_to_generate):
        tx = np.reshape(tx, (1, words_to_read, embedding_size))

        if (j == 0):
            print("loss : ", sess.run([loss], feed_dict={X: tx, Y: actualY, K.learning_phase(): 0}))

        predictedWord = model.predict(tx)
        '''
        predictedWord = closestEmbedding(predictedWord)
        predictedWord2 = embeddings[predictedWord]
        predictedWord2 = np.reshape(predictedWord2,[1,embedding_size])
        '''

        predictedWord = np.asarray(predictedWord).astype('float64')
        predictedWord = np.reshape(predictedWord, (vocab_size))
        # temperate
        # temperature = 0.6
        # predictedWord = np.log(predictedWord) / temperature
        # pw = np.exp(predictedWord)
        # predictedWord = pw / np.sum(pw)
        # pm = np.random.multinomial(1, predictedWord, 1)
        # pm = np.reshape(pm,(vocab_size))
        pm = predictedWord # either this or temp
        pw = np.exp(pm)
        pm = pw / np.sum(pw)
        predictedWord = np.argmax(pm)
        #
        # to not get UNK as prediction
        if (predictedWord==0):
            predictedWord = np.argsort(pm)[-2]
        #
        predictedWord2 = one_hot(predictedWord)
        predictedWord2 = np.reshape(predictedWord2,(1,vocab_size))
        tx = np.reshape(tx, (words_to_read, embedding_size))

        e = embeddings[predictedWord]
        e = np.reshape(e, (1, embedding_size))
        tx = np.append(tx, e, axis=0)
        tx = tx[1:]
        print(int2word[predictedWord],' ', end='')
        # tx = np.reshape(tx,(1,words_to_read,embedding_size))

# print (len(data2))  # - 106989 ...106932 106983
batch_size = int(len(data2)/213)

with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 51):
        print("\n\n Epoch ", epoch, "/500")
        for i in range(0,(len(data2) - words_to_read)-batch_size,batch_size):
            x = np.array([])
            y = np.array([])
            for j in range(i,i+batch_size,1):
                # print ('error at ?',i,j)
                x_, y_ = newXY(j)
                x = np.append(x,x_)
                y = np.append(y,y_)
            # X = tf.reshape(X,[1,words_to_read,embedding_size])
            x = np.reshape(x, (batch_size, words_to_read, embedding_size))
            y = np.reshape(y, (batch_size, vocab_size))
            train_step.run(feed_dict={X: x, Y: y, K.learning_phase(): 1})
            # model.fit(x, y, batch_size=batch_size)
        # if (i%1000==0):
        # test model
        generateTest()
        model.save_weights("rapWeights.h5")

print("Training Finished..........")
