#from __future__ import print_function
#import tensorflow as tf
import numpy as np
#import tqdm
from utils import dataset_input, data_prepro, batch_generator
#import tf.nn.rnn_cell.LSTMCell as LSTMCell
#import tf.nn.bidirectional_dynamic_rnn as BiRNN
from Embedding import Embedding


#Data preprocess
seq_maxlen = 5000

(train_data, y_train_all, test_data, y_test) = dataset_input()
x_train_list = data_prepro(train_data)
x_test_list = data_prepro(test_data)


#Embedding: each squence is a 300*5000 matrix
embed_size = 300
split_frac = 0.9

Embed = Embedding()
x_train_all = Embed.wordToVec(x_train_list)
x_test = Embed.wordToVec(x_test_list)

split_index = int(split_frac * len(x_train_all))
x_train, x_val = x_train_all[:split_index], x_train_all[split_index:] 
y_train, y_val = y_train_all[:split_index], y_train_all[split_index:]


#Build the graph
tf.reset_default_graph()
##Input
with tf.name_scope('Input'):
    batch_ph = tf.placeholder(tf.float32, [embed_size, seq_maxlen], name='batch_placeholder')
    output_ph = tf.placeholder(tf.int32,[None, y_train.shape[1]], name='output_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_placeholder')

'''
##Embedding
embed_size = 300

with tf.name_scope('Embedding Layer'):
    embedding = tf.Variable(tf.random_uniform([vocabulary_size, embed_size], -1, 1))
    embeded = tf.nn.embedding_lookup(embedding, batch_ph)
    tf.summary.histogram('embedding', embedding)
'''

##RNN layers
lstm_size = 128
lstm_layers = 2

with tf.name_scope('Multi Bi-LSTM Layers'):
    output = batch_ph
    for n in range(lstm_layers):
        cell_fw = LSTMCell(lstm_size)
        cell_bw = LSTMCell(lstm_size)
        (output_fw, output_bw), final_state = BiRNN(cell_fw, cell_bw, output)
        output = tf.divide(tf.add(output_fw, output_bw), 2)

    tf.summary.histogram('RNN_output', output)

##Dropout
    drop = tf.nn.dropout(output,keep_prob_ph)

##FC layers
with tf.name_scope('Fully connected Layers'):
    prediction = tf.contrib.layers.fully_connected(drop, 6, activation_fn=tf.nn.sigmoid)
    tf.summary.histogram('Prediction', prediction)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output_ph, logits=prediction))
    tf.summary.scalar('loss', loss)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
correct_pred = tf.equal(tf.cast(tf.round(prediction), tf.int32), output_ph)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
val_writer = tf.summary.FileWriter('./logdir/val', accuracy.graph)
saver = tf.train.Saver()


#Training and Evaluation
epoch = 100
keep_prop = 0.8
batch_size = 128
iteration = np.int32(np.round(x_train.shape[0] / batch_size))

with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    print('start training...')
    for m in range(epoch):
        print('Epoch: {} start!'.format(m + 1))
        for n in tqdm(range(iteration)):
            (x_batch, y_batch) = batch_generator(x_train, y_train, batch_size)
            loss_train,  _, summary = sess.run([loss, optimizer, merged],
                                              feed_dict={batch_ph: x_batch,
                                              output_ph: y_batch,
                                              keep_prob_ph: keep_prop})
            train_writer.add_summary(summary, n+m*iteration)
            #Each 100 iterations, run a valadition
            if (n + 1) % 100 == 0:
                val_acc = 0
                for k in range(100):
                    (x_batch, y_batch) = batch_generator(x_val, y_val, batch_size)
                    accuracy_val, summary = sess.run([accuracy, merged],
                                                    feed_dict={batch_ph: x_batch,
                                                    output_ph: y_batch,
                                                    keep_prob_ph: 1})
                    val_acc += accuracy_val
                print('Iteration: {}'.format(n+1),
                      'Train loss: {:.3f}'.format(loss_train),
                      'Val accuracy: {:.3f}'.format(val_acc/100))
                val_writer.add_summary(summary, n+m*iteration)

        print('Epoch: {} finished!'.format(m+1), 'Train loss: {}'.format(loss_train))
        saver.save(sess, './model')
    print('**********************TRAINING FINISHED!**********************')
    saver.save(sess, './model/final.ckpt')

train_writer.close()
val_writer.close()

#Testing

with tf.Session as sess:
    saver.restore(sess, './model/final.ckpt')
    test_acc = sess.run(accuracy,
                       feed_dict={batch_ph: x_test,
                       output_ph: y_test,
                       keep_prob_ph: 1})
    print('Test Accuracy: {:.3f}'.format(test_acc))

