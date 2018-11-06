import tensorflow as tf
import numpy as np
#import tqdm
from utils import dataset_input, data_prepro, batch_generator
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn import bidirectional_dynamic_rnn as BiRNN
from tensorflow.contrib.rnn import MultiRNNCell
from Embedding import Embedding

seq_maxlen = 100

(train_data, y_train_all, test_data, y_test) = dataset_input()
x_test_list = data_prepro(test_data)

embed_size = 25

Embed = Embedding()
x_test = Embed.wordsToVec(x_test_list)


batch_size = 128
(x_batch, y_batch) = batch_generator(x_test, y_test, batch_size)

with tf.Session() as sess:
    saver.restore(sess, './model/final.ckpt')
    iteration = 1200
    test_acc = 0

    for i in range(iteration):
        test_accuracy = sess.run(accuracy,
                        feed_dict={batch_ph: x_test,
                        output_ph: y_test,
                        keep_prob_ph: 1})
        test_acc+= test_accuracy

    print('Test Accuracy: {:.3f}'.format(test_acc/iteration))