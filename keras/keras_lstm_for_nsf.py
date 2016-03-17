from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import cPickle
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.models import model_from_json
import theano
from keras.layers.recurrent import LSTM





batch_size = 50
embedding_dims = 300
nb_filter = 300
filter_length = 3
hidden_dims = 200
nb_epoch = 15
nb_classes = 4
lstm_output_size = 200

x = cPickle.load(open("mr.p","rb"))
revs, W, W2, word_idx_map, vocab, maxlen = x[0], x[1], x[2], x[3], x[4], x[5]

maxlen = int(maxlen)

max_features = len(word_idx_map)+1
print('max_features : ',max_features)
print("maxlen : ", maxlen)

def get_idx_from_sent(sent, word_idx_map, max_l=51, k = 300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """

    x = []
  
    words = sent
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    return x

print('Loading data...')
X_train, X_test = [], []
Y_train, Y_test = [], []
X_raw =[]
X= []

for rev in revs:
        X_raw.append(rev["text"])
        sent = get_idx_from_sent(rev["text"], word_idx_map, maxlen, embedding_dims, filter_length)
        X.append(sent)

        if rev["split"]==0:
            X_test.append(sent)
            Y_test.append(rev["y"])
        else:
            X_train.append(sent)
            Y_train.append(rev["y"])



print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train,maxlen=maxlen)
X_test = sequence.pad_sequences(X_test,maxlen=maxlen)
X = sequence.pad_sequences(X,maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)



print('Build model...')
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


embedding_weights = np.zeros((max_features+1,embedding_dims))
for word,index in word_idx_map.items():
    embedding_weights[index,:] = W[index]




model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(input_dim=max_features+1, output_dim=embedding_dims, weights=[embedding_weights],input_length=maxlen))

model.add(LSTM(lstm_output_size))
model.add(Dropout(0.5))

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims,W_regularizer=l2(0.01)))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adagrad',
              class_mode='categorical')
hist = model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, Y_test))

f = open("log_lstm.txt","w")

h = hist.history
for i in range(nb_epoch):
    his = "epoch : " + str(i+1) +" : acc : "+  str(h["acc"][i]) + " , loss : " + str(h["loss"][i]) + \
    " , val_acc : " + str(h["val_acc"][i]) + " , val_loss : "+  str(h["val_loss"][i])
    f.writelines(his+'\n')


score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
f.write(str('\n' + "Test score : " + str(score[0]) +'\n'))
f.write(str("Test accuracy : " + str(score[1]) +'\n'))
f.close()
