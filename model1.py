import tensorflow as tf
from tensorflow.contrib import rnn
import numpy

def feed_forward(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forward', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff
def feed_forward2(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forward2', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff
def feed_forward3(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forward3', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff
def feed_forward4(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forward4', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff
def feed_forwardd(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forwardd', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff
def feed_forwardd2(x, num_hiddens, activation=None, reuse=False):
    with tf.variable_scope('feed-forwardd2', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff

def linear(x, num_hiddens=None, reuse=False):
    if num_hiddens is None:
        num_hiddens = x.get_shape().as_list()[-1]
    # with tf.variable_scope('linear'):
    linear_layer = tf.layers.dense(x, num_hiddens)
    return linear_layer


def dropout(x, is_training, rate=0.2):
    return tf.layers.dropout(x, rate, training=tf.convert_to_tensor(is_training))


def residual(x_in, x_out, reuse=False):
    with tf.variable_scope('residual', reuse=reuse):
        res_con = x_in + x_out
    return res_con

def stacked_multihead_attention_d(x,y, num_blocks, num_heads, use_residual, is_training, reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention_d', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attentiond(x, y, y, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forwardd(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions


def multihead_attentiond(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attentiond', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forwardd(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
        return output, attentions

def stacked_multihead_attention_d2(x,y, num_blocks, num_heads, use_residual, is_training, reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attentiond2(x, y, y, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forwardd2(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
        return x, attentions
    
def multihead_attentiond2(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attentiond2', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forwardd2(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
        return output, attentions

def stacked_multihead_attention(x, num_blocks, num_heads, use_residual, is_training, reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attention(x, x, x, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forward(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions

def multihead_attention(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attention', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forward(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
        
    return output, attentions

def stacked_multihead_attention2(x, num_blocks, num_heads, use_residual, is_training, reuse=False):
    print(x)
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention2', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attention2(x, x, x, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forward2(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions

def multihead_attention2(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attention2', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forward2(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
        
    return output, attentions

def stacked_multihead_attention3(x, num_blocks, num_heads, use_residual, is_training, reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention3', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attention3(x, x, x, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forward3(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions

def stacked_multihead_attention4(x, num_blocks, num_heads, use_residual, is_training, reuse=False):
    num_hiddens = x.get_shape().as_list()[-1]
    with tf.variable_scope('stacked_multihead_attention4', reuse=reuse):
        for i in range(num_blocks):
            with tf.variable_scope('multihead_block_{}'.format(i), reuse=reuse):
                x, attentions = multihead_attention4(x, x, x, use_residual, is_training, num_heads=num_heads, reuse=reuse)
                x = feed_forward4(x, num_hiddens=num_hiddens, activation=tf.nn.relu, reuse=reuse)
    return x, attentions

def multihead_attention4(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attention4', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forward4(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
        
    return output, attentions

def multihead_attention3(queries, keys, values, use_residual, is_training, num_units=None, num_heads=8, reuse=False):
    with tf.variable_scope('multihead-attention3', reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        Q = linear(queries)
        K = linear(keys)
        V = linear(values)

        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        Q_K_V, attentions = scaled_dot_product_attention(Q, K, V)
        Q_K_V = dropout(Q_K_V, is_training)
        Q_K_V_ = tf.concat(tf.split(Q_K_V, num_heads, axis=0), axis=2)

        output = feed_forward3(Q_K_V_, num_units, reuse=reuse)

        if use_residual:
            output = residual(output, queries, reuse=reuse)
        # output = normalization(output)
        
    return output, attentions

def scaled_dot_product_attention(queries, keys, values, model_size=None, reuse=False):
    if model_size is None:
        model_size = tf.to_float(queries.get_shape().as_list()[-1])

    with tf.variable_scope('scaled_dot_product_attention', reuse=reuse):
        keys_T = tf.transpose(keys, [0, 2, 1])
        Q_K = tf.matmul(queries, keys_T) / tf.sqrt(model_size)
        attentions = tf.nn.softmax(Q_K)
        scaled_dprod_att = tf.matmul(attentions, values)
    return scaled_dprod_att, attentions

class Model(object):
    def __init__(self, vocabulary_size, num_class, args):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden
        self.is_training = True

        self.x = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.xcola=tf.placeholder(tf.int32, [None, args.max_document_len])
        self.xnli1=tf.placeholder(tf.int32, [None, args.max_document_len])
        self.xnli2=tf.placeholder(tf.int32, [None, args.max_document_len])
        self.xsts1=tf.placeholder(tf.int32, [None, args.max_document_len])
        self.xsts2 = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.lm_y = tf.placeholder(tf.int32, [None, args.max_document_len])
        self.clf_y = tf.placeholder(tf.int32, [None])
        self.clf_nli = tf.placeholder(tf.int32, [None])
        self.clf_sts = tf.placeholder(tf.int32, [None])
        self.clf_cola = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32, [])

        self.x_len = tf.reduce_sum(tf.sign(self.x), 1)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(embeddings, self.x)
            self.xcola_emb = tf.nn.embedding_lookup(embeddings, self.xcola)
            self.xnli1_emb = tf.nn.embedding_lookup(embeddings, self.xnli1)
            self.xnli2_emb = tf.nn.embedding_lookup(embeddings, self.xnli1)
            self.sts1_emb = tf.nn.embedding_lookup(embeddings, self.xsts1)
            self.sts2_emb = tf.nn.embedding_lookup(embeddings, self.xsts2)
           
        with tf.name_scope("rnn"):
            #self.x_trans=stacked_multihead_attention(self.x_emb,num_blocks=3,num_heads=5,use_residual=False,is_training=self.is_training)
            cell = rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)])
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, self.x_emb, sequence_length=self.x_len, dtype=tf.float32)
            
        with tf.name_scope("base_transformer"):
            self.base_cola,_=stacked_multihead_attention(self.xcola_emb,num_blocks=3,num_heads=5,use_residual=False,is_training=self.is_training)
            self.base_nli1,_=stacked_multihead_attention(self.xnli1_emb,num_blocks=3,num_heads=5,use_residual=False,is_training=self.is_training,reuse=True)
            self.base_nli2,_=stacked_multihead_attention(self.xnli2_emb,num_blocks=3,num_heads=5,use_residual=False,is_training=self.is_training,reuse=True)
            self.base_sts1,_=stacked_multihead_attention(self.sts1_emb,num_blocks=3,num_heads=5,use_residual=False,is_training=self.is_training,reuse=True)
            self.base_sts2,_=stacked_multihead_attention(self.sts2_emb,num_blocks=3,num_heads=5,use_residual=False,is_training=self.is_training,reuse=True)
            
        with tf.name_scope("lm"):
            self.lm_logits = tf.layers.dense(rnn_outputs, vocabulary_size)
            
        with tf.name_scope("cola"):
            self.transform_output,_=stacked_multihead_attention4(self.base_cola,num_blocks=2,num_heads=3,use_residual=False,is_training=self.is_training)
            self.meancola=tf.reduce_sum(self.transform_output, axis=1)
            self.clf_logitscola = tf.layers.dense(self.meancola, 6)

        with tf.name_scope("nli"):
            #rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, args.max_document_len * self.num_hidden])
            self.transform_output21,_=stacked_multihead_attention2(self.base_nli1,num_blocks=2,num_heads=3,use_residual=False,is_training=self.is_training)
            self.transform_output22,_=stacked_multihead_attention2(self.base_nli2,num_blocks=2,num_heads=3,use_residual=False,is_training=self.is_training,reuse=True)
            self.transform_output23,_=stacked_multihead_attention_d(self.base_nli1,self.base_nli2,num_blocks=1,num_heads=3,use_residual=False,is_training=self.is_training)
            self.meannli=tf.reduce_sum(self.transform_output23, axis=1)
            self.clf_logitsnli = tf.layers.dense(self.meannli, 3)
            
        with tf.name_scope("clf-output"):
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, args.max_document_len * self.num_hidden])
            self.clf_logits = tf.layers.dense(rnn_outputs_flat, num_class)
            self.clf_predictions = tf.argmax(self.clf_logits, -1, output_type=tf.int32)
            
        with tf.name_scope("sts"):
            #rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, args.max_document_len * self.num_hidden])
            self.transform_output31,_=stacked_multihead_attention3(self.base_sts1,num_blocks=2,num_heads=3,use_residual=False,is_training=self.is_training)
            #self.clf_logits = tf.layers.dense(self.transform_output2, num_class)
            self.transform_output32,_=stacked_multihead_attention3(self.base_sts2,num_blocks=2,num_heads=3,use_residual=False,is_training=self.is_training,reuse=True)
            self.transform_output33,_=stacked_multihead_attention_d2(self.base_sts1,self.base_sts2,num_blocks=1,num_heads=3,use_residual=False,is_training=self.is_training)
            self.meansts=tf.reduce_sum(self.transform_output33, axis=1)
            self.clf_logitssts= tf.layers.dense(self.meansts, 3)
            
        with tf.name_scope("loss"):
            self.lm_loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.lm_logits,
                targets=self.lm_y,
                weights=tf.sequence_mask(self.x_len, args.max_document_len, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)
            
            self.clf_loss_nli = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.clf_logitsnli, labels=self.clf_nli))
            
            self.clf_loss_sts = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.clf_logitssts, labels=self.clf_sts))
            
            self.clf_loss_cola = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.clf_logitscola, labels=self.clf_cola))
           
            self.total_loss = self.lm_loss + self.clf_loss_nli + self.clf_loss_sts + self.clf_loss_cola 
                #tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.clf_logits, labels=self.clf_y))
                
        with tf.name_scope("clf-accuracy"):
            correct_predictions = tf.equal(self.clf_predictions, self.clf_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    def make_cell(self):
        cell = rnn.BasicLSTMCell(self.num_hidden)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=0.5)
        return cell
