import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
from utils import dataset_input, data_prepro, batch_generator
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn import bidirectional_dynamic_rnn as BiRNN
#from Embedding import Embedding
from Attention import Attention

EMBEDDING_FILE = 'glove.twitter.27B.25d.txt'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

embed_size = 25 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
split_frac = 0.9

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train_all = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
x_train_all = pad_sequences(list_tokenized_train, maxlen=maxlen)
x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

split_index = int(split_frac * len(x_train_all))
x_train, x_val = x_train_all[:split_index], x_train_all[split_index:]
y_train, y_val = y_train_all[:split_index], y_train_all[split_index:]

def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

emb_mean, emb_std = 0.020940498, 0.6441043

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Build the graph
tf.reset_default_graph()
##Input
with tf.name_scope('Input'):
    batch_ph = tf.placeholder(tf.int32, [None, maxlen], name='batch_placeholder')
    output_ph = tf.placeholder(tf.float32, [None, 6], name='output_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_placeholder')
    keep_prob_ph_rnn = tf.placeholder(tf.float32, name='keep_prob_placeholder_rnn')

with tf.variable_scope('Embedding_layer'):
    W = tf.get_variable(name="Embedding_matrix", shape=embedding_matrix.shape, initializer=tf.constant_initializer(embedding_matrix), trainable=False)
    embeded = tf.nn.embedding_lookup(W, batch_ph)

##RNN layers
lstm_size = 32
lstm_layers = 3

output = embeded
'''
for i in range(lstm_layers):
    with tf.variable_scope('BiLSTM_Layer_{}'.format(i)):
        lstm_fw = LSTMCell(lstm_size) #, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2)
        lstm_bw = LSTMCell(lstm_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw, output_keep_prob=keep_prob_ph_rnn)
        cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw, output_keep_prob=keep_prob_ph_rnn)

        (output_fw, output_bw), final_state = BiRNN(cell_fw, cell_bw, output, dtype=tf.float32)
        output = tf.concat((output_fw, output_bw), 2)
'''

##Attention + Dropout
with tf.variable_scope('BiLSTM_Layer_{}'.format(lstm_layers)):
    lstm_fw = LSTMCell(lstm_size)
    lstm_bw = LSTMCell(lstm_size)
    (output_fw, output_bw), final_state = BiRNN(lstm_fw, lstm_bw, output, dtype=tf.float32)
    output = tf.concat((output_fw, output_bw), 2)
    attention = Attention(output)
    drop = tf.nn.dropout(attention, keep_prob_ph)
    tf.summary.histogram('RNN_output', output)


##FC layers
with tf.name_scope('Fully_connected_Layers_0'):
    fc_output = tf.contrib.layers.fully_connected(drop, 64, activation_fn=tf.nn.relu)
    tf.summary.histogram('fc_output', fc_output)

with tf.name_scope('Fully_connected_Layers_1'):
    prediction = tf.contrib.layers.fully_connected(fc_output, 6, activation_fn=tf.nn.sigmoid)
    tf.summary.histogram('Prediction', prediction)

with tf.name_scope('Loss'):
    loss = tf.losses.mean_squared_error(output_ph, prediction)
    tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
#( ,6) == ( ,6)
correct_pred = tf.reduce_all(tf.equal(tf.cast(tf.round(prediction), tf.int32), tf.cast(output_ph, tf.int32)), axis=1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
val_writer = tf.summary.FileWriter('./logdir/val', accuracy.graph)
saver = tf.train.Saver()

#Training and Evaluation
epoch = 2
batch_size = 32
batch_size_val = 64

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print('start training...')

    for m in range(epoch):

        print('Epoch: {} start!'.format(m + 1))
        gen_batch_train = batch_generator(x_train, y_train, batch_size)
        iteration = np.int(y_train.shape[0] / batch_size)
        for n in range(iteration):
            (x_batch, y_batch) = next(gen_batch_train)
            loss_train,  _, summary = sess.run([loss, optimizer, merged],
                                               feed_dict={batch_ph: x_batch,
                                                          output_ph: y_batch,
                                                          keep_prob_ph: 0.9,
                                                          keep_prob_ph_rnn: 0.9})
            train_writer.add_summary(summary, n+m*iteration)
            #Each 100 iterations, run a valadition
            if (n + 1) % 100 == 0:
                val_acc = 0
                gen_batch_val = batch_generator(x_val, y_val, batch_size_val)
                itr_val = np.int(y_val.shape[0] / batch_size_val)
                for k in range(itr_val):
                    (x_batch, y_batch) = next(gen_batch_val)
                    accuracy_val, summary = sess.run([accuracy, merged],
                                                     feed_dict={batch_ph: x_batch,
                                                                output_ph: y_batch,
                                                                keep_prob_ph: 1,
                                                                keep_prob_ph_rnn: 1})
                    val_acc += accuracy_val
                print('Iteration: {}'.format(n+1),
                      'Train loss: {:.5f}'.format(loss_train),
                      'Val accuracy: {:.8f}'.format(val_acc/itr_val))
                val_writer.add_summary(summary, n+m*iteration)

        print('Epoch: {} finished!'.format(m+1), 'Train loss: {:.3f}'.format(loss_train))
        saver.save(sess, './model/intermediate.ckpt')
    print('**********************TRAINING FINISHED!**********************')
    saver.save(sess, './model/final.ckpt')

train_writer.close()
val_writer.close()

'''
#Testing
batch_s = 64

with tf.Session() as sess:
    saver.restore(sess, './model/final.ckpt')
    iteration = np.int(y_test.shape[0] / batch_s)
    gen_batch_test = batch_generator(x_test, y_test, batch_s)
    test_acc = 0

    for i in range(iteration):
        (x_batch, y_batch) = next(gen_batch_test)
        test_accuracy = sess.run(accuracy,
                                 feed_dict={batch_ph: x_batch,
                                            output_ph: y_batch,
                                            keep_prob_ph: 1,
                                            keep_prob_ph_rnn: 1})
        test_acc += test_accuracy

    print('Test Accuracy: {:.3f}'.format(test_acc/iteration))
'''
