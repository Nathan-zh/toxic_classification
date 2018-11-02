from __future__ import print_function
import numpy as np
import pandas as pd

def dataset_input():
    #read training data
    df_train = pd.read_csv('train.csv')
    
    df_y = df_train[['toxic']] #, 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    train_label = df_y.values
    
    df_x = df_train[['comment_text']]
    train_data = df_x.values.tolist()
    
    #reading test data
    df_test = pd.read_csv('test.csv')
    df_test1 = pd.read_csv('test_labels.csv')
    
    df_y1 = df_test1[['toxic']] #, 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    idx0 = df_y1.index[df_y1['toxic']==0].values
    idx1 = df_y1.index[df_y1['toxic']==1].values
    idx = np.concatenate((idx0, idx1))
    test_label = df_y1.values[idx]
    
    df_x1 = df_test[['comment_text']]
    test_data1 = df_x1.values.tolist()
    test_data =  [test_data1[i] for i in idx]
    
    return (train_data, train_label, test_data, test_label)


def data_prepro(x_input):

    #delete punctuation and duplicated space
    whitelist = set('abcdefghijklmnopqrstuvwxyz 1234567890')
    x_output = []
    for m in x_input:
        all_text = ''.join(filter(whitelist.__contains__, m[0].lower()))
        text = ' '.join(all_text.split())
        x_output.append(text)
    
    return x_output


def batch_generator(x_input, y_input, batch_size):

    count = y_input.shape[0]
    idx = np.arange(count)
    np.random.shuffle(idx)
    x_batch, y_batch = [x_input[i] for i in idx[:batch_size]], y_input[idx[:batch_size]]

    return (x_batch, y_batch)