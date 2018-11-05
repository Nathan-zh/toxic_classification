import tensorflow as tf
import numpy as np
#import tqdm
from utils import dataset_input, data_prepro, batch_generator
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn import bidirectional_dynamic_rnn as BiRNN
from tensorflow.contrib.rnn import MultiRNNCell
from Embedding import Embedding

seq_maxlen = 5000

(train_data, y_train_all, test_data, y_test) = dataset_input()
#x_train_list = data_prepro(train_data)
x_test_list = data_prepro(test_data)


#Embedding: each squence is a 300*5000 matrix
embed_size = 300
split_frac = 0.9

Embed = Embedding()
#x_train_all = Embed.wordsToVec(x_train_list)
x_test = Embed.wordsToVec(x_test_list)

#split_index = int(split_frac * len(x_train_all))
#x_train, x_val = x_train_all[:split_index], x_train_all[split_index:]
#y_train, y_val = y_train_all[:split_index], y_train_all[split_index:]


batch_size = 32
(x_batch, y_batch) = batch_generator(x_test, y_test, batch_size)

with tf.Session() as sess:
    saver.restore(sess, './model')
    test_acc = sess.run(accuracy,
                       feed_dict={batch_ph: x_test,
                       output_ph: y_test,
                       keep_prob_ph: 1})
    print('Test Accuracy: {:.3f}'.format(test_acc))