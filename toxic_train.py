#from __future__ import print_function
import tensorflow as tf
import numpy as np
#import tqdm
from utils import dataset_input, data_prepro, batch_generator
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn import bidirectional_dynamic_rnn as BiRNN
#from tensorflow.contrib.rnn import MultiRNNCell
from Embedding import Embedding
from Attention import Attention


#Data preprocess
seq_maxlen = 100

(train_data, y_train_all, test_data, y_test) = dataset_input()
x_train_list = data_prepro(train_data)
x_test_list = data_prepro(test_data)


#Embedding: each squence is a 300*5000 matrix
embed_size = 25
split_frac = 0.9

Embed = Embedding()
x_train_all = Embed.wordsToVec(x_train_list)
x_test = Embed.wordsToVec(x_test_list)

split_index = int(split_frac * len(x_train_all))
x_train, x_val = x_train_all[:split_index], x_train_all[split_index:] 
y_train, y_val = y_train_all[:split_index], y_train_all[split_index:]


#Build the graph
tf.reset_default_graph()
##Input
with tf.name_scope('Input'):
    batch_ph = tf.placeholder(tf.float32, [None, seq_maxlen, embed_size], name='batch_placeholder')
    output_ph = tf.placeholder(tf.float32, [None, 6], name='output_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_placeholder')

'''
##Embedding
embed_size = 25

with tf.name_scope('Embedding Layer'):
    embedding = tf.Variable(tf.random_uniform([vocabulary_size, embed_size], -1, 1))
    embeded = tf.nn.embedding_lookup(embedding, batch_ph)
    tf.summary.histogram('embedding', embedding)
'''

##RNN layers
lstm_size = 32
lstm_layers = 2


output = batch_ph
for i in range(lstm_layers):
    with tf.variable_scope('BiLSTM_Layer_{}'.format(i)):
        lstm_fw = LSTMCell(lstm_size)
        lstm_bw = LSTMCell(lstm_size)
        #cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_fw, output_keep_prob=keep_prob_ph)
        #cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_bw, output_keep_prob=keep_prob_ph)

        (output_fw, output_bw), final_state = BiRNN(lstm_fw, lstm_bw, output, dtype=tf.float32)
        output = tf.concat((output_fw, output_bw), 2)

tf.summary.histogram('RNN_output', output)

##Attention + Dropout
attention = Attention(output)
drop = tf.nn.dropout(attention, keep_prob_ph)

##FC layers
with tf.name_scope('Fully_connected_Layers'):
    prediction = tf.contrib.layers.fully_connected(drop, 6, activation_fn=tf.nn.sigmoid)
    tf.summary.histogram('Prediction', prediction)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_ph, logits=prediction))
    tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
#( ,6) == ( ,6)
correct_pred = tf.reduce_all(tf.equal(tf.cast(tf.round(prediction), tf.int32), tf.cast(output_ph, tf.int32)), axis=1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
val_writer = tf.summary.FileWriter('./logdir/val', accuracy.graph)
saver = tf.train.Saver()


#Training and Evaluation
epoch = 10
keep_prop = 0.5
batch_size = 16
batch_size_val = 64

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print('start training...')

    for m in range(epoch):

        print('Epoch: {} start!'.format(m + 1))
        gen_batch_train = batch_generator(x_train, y_train, batch_size)
        iteration = 50 #np.int(y_train.shape[0] / batch_size)
        for n in range(iteration):
            (x_batch, y_batch) = next(gen_batch_train)
            loss_train,  _, summary = sess.run([loss, optimizer, merged],
                                              feed_dict={batch_ph: x_batch,
                                              output_ph: y_batch,
                                              keep_prob_ph: keep_prop})
            train_writer.add_summary(summary, n+m*iteration)
            #Each 100 iterations, run a valadition
            if (n + 1) % 10 == 0:
                val_acc = 0
                gen_batch_val = batch_generator(x_val, y_val, batch_size_val)
                itr_val = np.int(y_val.shape[0] / batch_size_val)
                for k in range(itr_val):
                    (x_batch, y_batch) = next(gen_batch_val)
                    accuracy_val, summary = sess.run([accuracy, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                    output_ph: y_batch,
                                                    keep_prob_ph: 1})
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
                                keep_prob_ph: 1})
        test_acc+= test_accuracy

    print('Test Accuracy: {:.3f}'.format(test_acc/iteration))

