import tensorflow as tf


# size: batch*seq*input_dim
def Attention(inputs):

    attention_size = 32 #output_dim
    inputs_dim = inputs.shape[2].value
    w_omega = tf.get_variable("W", shape=[inputs_dim, attention_size], initializer=tf.random_normal()) # input_dim*output_dim
    b_omega = tf.get_variable("b", shape=[attention_size], initializer=tf.random_normal()) # output_dim
    u_omega = tf.get_variable("u", shape=[attention_size], initializer=tf.random_normal()) # output_dim

    v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega) # batch*seq*output_dim
    vu = tf.tensordot(v, u_omega, axes=1, name='vu') # batch*seq
    alphas = tf.nn.softmax(vu, name='alphas') # batch*seq
    temp = tf.expand_dims(alphas, -1) # batch*seq*1
    output = tf.reduce_sum(inputs * temp, 1) # size: batch*input_dim

    return output
