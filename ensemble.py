import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from Attention_keras import Attention


def data_input(EMBEDDING_FILE, embed_size, max_features, maxlen):

    TRAIN_DATA_FILE = 'train.csv'
    TEST_DATA_FILE = 'test.csv'

    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)

    sentences_train = train["comment_text"].values.tolist()
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_t = train[list_classes].values
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

    def get_coefs(word, *arr):
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

    df_test_label = pd.read_csv('test_labels.csv')
    df_y1 = df_test_label[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    idx0 = df_y1.index[df_y1['toxic'] == 0].values
    idx1 = df_y1.index[df_y1['toxic'] == 1].values
    idx = np.concatenate((idx0, idx1))
    y_test = df_y1.values[idx]
    X_test = X_te[idx]

    return X_t, y_t, X_test, y_test, embedding_matrix


def compile_and_train(model, X_t, y_t, num_epochs, num_model):

    MODEL_PATH = './keras_model/model{}/'.format(num_model)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir=MODEL_PATH+'log/', histogram_freq=0, write_graph=True,
                              write_images=True)
    model.fit(X_t, y_t, batch_size=32, epochs=num_epochs, validation_split=0.1, callbacks=[tensorboard])

    model.save_weights(MODEL_PATH + 'model.h5')
    print("Saved weights to disk %s" % MODEL_PATH)

    plot_model(model, to_file=MODEL_PATH + 'graph.png')
    print("Saved graph to disk %s" % MODEL_PATH)

    return None

def evaluate(model, X_test, y_test):

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(x=X_test, y=y_test, batch_size=64, verbose=1)

    return score


def predict(model, X_test):

    y_pred = model.predict(x=X_test, batch_size=64, verbose=1)

    return y_pred


def Model1(maxlen, max_features, embed_size, embedding_matrix):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(GRU(64, return_sequences=True, return_state=False, dropout=0.5,
                          recurrent_dropout=0.1))(x)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def Model2(maxlen, max_features, embed_size, embedding_matrix):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(GRU(64, return_sequences=True, return_state=False, dropout=0.5,
                          recurrent_dropout=0.1))(x)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def Model3(maxlen, max_features, embed_size, embedding_matrix):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(GRU(64, return_sequences=True, return_state=False, dropout=0.5,
                          recurrent_dropout=0.1))(x)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def Model4(maxlen, max_features, embed_size, embedding_matrix):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(GRU(64, return_sequences=True, return_state=False, dropout=0.5,
                          recurrent_dropout=0.1))(x)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def Model5(maxlen, max_features, embed_size, embedding_matrix):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(GRU(64, return_sequences=True, return_state=False, dropout=0.5,
                          recurrent_dropout=0.1))(x)
    #x = GlobalAveragePooling1D()(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


# Model1 training and weights saving / evaluation
EMBEDDING_FILE = 'glove.6B.50d.txt'
embed_size = 50
max_features = 20000
maxlen = 100
X_t, y_t, X_test, y_test, embedding_matrix = data_input(EMBEDDING_FILE, embed_size, max_features, maxlen)
model1 = Model1(maxlen, max_features, embed_size, embedding_matrix)
model1.load_weights('./keras_model/model1/model.h5')
#compile_and_train(model1, X_t, y_t, num_epochs=2, num_model=1)
score1 = evaluate(model1, X_test, y_test) #[loss, accuracy]
print('********* Model 1 test accuracy is %.4f *********' % score1[1])
y_pred1 = predict(model1, X_test)

# Model2 training and weights saving / evaluation
EMBEDDING_FILE = 'glove.twitter.27B.25d.txt'
embed_size = 25
max_features = 20000
maxlen = 100
X_t, y_t, X_test, y_test, embedding_matrix = data_input(EMBEDDING_FILE, embed_size, max_features, maxlen)
model2 = Model2(maxlen, max_features, embed_size, embedding_matrix)
model2.load_weights('./keras_model/model2/model.h5')
#compile_and_train(model2, X_t, y_t, num_epochs=2, num_model=2)
score2 = evaluate(model2, X_test, y_test) #[loss, accuracy]
print('********* Model 2 test accuracy is %.4f *********' % score2[1])
y_pred2 = predict(model2, X_test)

# Model3 training and weights saving / evaluation
EMBEDDING_FILE = 'glove.6B.300d.txt'
embed_size = 300
max_features = 20000
maxlen = 100
X_t, y_t, X_test, y_test, embedding_matrix = data_input(EMBEDDING_FILE, embed_size, max_features, maxlen)
model3 = Model3(maxlen, max_features, embed_size, embedding_matrix)
model3.load_weights('./keras_model/model3/model.h5')
#compile_and_train(model3, X_t, y_t, num_epochs=2, num_model=3)
score3 = evaluate(model3, X_test, y_test) #[loss, accuracy]
print('********* Model 3 test accuracy is %.4f *********' % score3[1])
y_pred3 = predict(model3, X_test)

'''
# Model4 training and weights saving / evaluation
EMBEDDING_FILE = 'glove.6B.50d.txt'
embed_size = 50
max_features = 20000
maxlen = 100
X_t, y_t, X_test, y_test, embedding_matrix = data_input(EMBEDDING_FILE, embed_size, max_features, maxlen)
model4 = Model4(maxlen, max_features, embed_size, embedding_matrix)
#model4.load_weights('./keras_model/model4/model.h5')
compile_and_train(model4, X_t, y_t, num_epochs=2, num_model=4)
score4 = evaluate(model4, X_test, y_test) #[loss, accuracy]
print('********* Model 4 test accuracy is %.4f *********' % score4[1])

# Model5 training and weights saving / evaluation
EMBEDDING_FILE = 'glove.6B.50d.txt'
embed_size = 50
max_features = 20000
maxlen = 100
X_t, y_t, X_test, y_test, embedding_matrix = data_input(EMBEDDING_FILE, embed_size, max_features, maxlen)
model5 = Model5(maxlen, max_features, embed_size, embedding_matrix)
#model5.load_weights('./keras_model/model5/model.h5')
compile_and_train(model5, X_t, y_t, num_epochs=2, num_model=5)
score5 = evaluate(model5, X_test, y_test) #[loss, accuracy]
print('********* Model 5 test accuracy is %.4f *********' % score5[1])
'''

multi_pred = (y_pred1 + y_pred2 + y_pred3) / 3
correct_pred = np.equal(np.int32(np.round(multi_pred)), y_test)
ensemble_acc = np.mean(correct_pred)
print('********* Ensemble model test accuracy is %.4f *********' % ensemble_acc)
