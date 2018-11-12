import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
#from keras import initializers, regularizers, constraints, optimizers, layers
from Attention_keras import Attention

EMBEDDING_FILE = 'glove.6B.50d.txt'  # glove.twitter.27B.25d.txt
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

sentences_train = train["comment_text"].values.tolist()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
sentences_test = test["comment_text"].values.tolist()


def data_prepro(x_input):
    # delete punctuation and duplicated space
    whitelist = set('abcdefghijklmnopqrstuvwxyz 1234567890')
    x_output = []
    for m in x_input:
        all_text = ''.join(filter(whitelist.__contains__, m.lower()))
        text = ' '.join(all_text.split())
        x_output.append(text)

    return x_output


list_sentences_train = data_prepro(sentences_train)
list_sentences_test = data_prepro(sentences_test)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list_sentences_train)
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(GRU(32, return_sequences=True, return_state=False, dropout=0.1,
                       recurrent_dropout=0.1))(x)
#x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
#x = Bidirectional(LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
#x = Attention(maxlen)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_t, y, batch_size=32, epochs=2, validation_split=0.1)

'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''

'''
y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('submission.csv', index=False)
'''