# -*- coding: UTF-8 -*-

import tensorflow as tf

import numpy as np
class LSTM_CRF():
    def __init__(self,config,is_training= 0,is_test = 0):
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
        self.batch_size = tf.Variable(config.batch_size, dtype=tf.int32, trainable=False)
        num_step = config.num_step
        class_num = config.class_num
        words_len = config.word_len
        embed_dim = config.embed_dim
        self.input_data = tf.placeholder(tf.float32, [None, num_step])
        self.target = tf.placeholder(tf.int32, [None,num_step])
        self.mask = tf.placeholder(tf.int32, [None,])
        self.seq_class = tf.placeholder(tf.int32, [None,config.seq_class])
        hidden_neural_size = config.hidden_neural_size
        hidden_layer_num = config.hidden_layer_num
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self.is_training =  tf.placeholder(tf.bool)
        self.trans_params = tf.get_variable("transitions",[class_num,class_num],tf.float32)
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        input_data = tf.reshape(self.input_data, [-1, config.vector_len])
        self.pad = tf.placeholder(tf.float32, [None, 1, embed_dim, 1], name='pad')
        l2_loss = tf.constant(0.01)
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding",[words_len,embed_dim],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            emb = tf.expand_dims(inputs, -1)
        pooled_concat = []
        reduced = np.int32(np.ceil((208) * 1.0 / 4))
        for i, filter_size in enumerate([3,4,5]):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.pad] * num_prio, 1)
                pad_post = tf.concat([self.pad] * num_post, 1)
                emb_pad = tf.concat([pad_prio, emb, pad_post], 1)

                filter_shape = [filter_size, embed_dim, 1, 32]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                conv = tf.nn.conv2d(emb_pad, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')

                h = tf.nn.relu(conv, name='relu')
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1],
                                        padding='SAME', name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, 32])
                pooled_concat.append(pooled)

        pooled_concat = tf.concat(pooled_concat, 2)
        pooled_concat = tf.nn.dropout(pooled_concat, self.keep_prob)

        cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_neural_size, forget_bias=1.0)
                                               for _ in range(hidden_layer_num)] , state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_neural_size, forget_bias=1.0)
                                               for _ in range(hidden_layer_num)], state_is_tuple=True)
        _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=pooled_concat,
            dtype=tf.float32,
            sequence_length=self.mask)

        (output_fw, output_bw), _ = _output
        output = tf.concat([output_fw, output_bw],2)
        output = tf.reshape(output,[-1,2 * hidden_neural_size])
        dense_size = 1024
        with tf.variable_scope("dense1"):
            W1 = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * hidden_neural_size, dense_size])

            b1 = tf.get_variable("b", shape=[dense_size],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            output = tf.matmul(output, W1) + b1
            output = tf.nn.relu(output)

        with tf.variable_scope("proj"):
            W2 = tf.get_variable("W", dtype=tf.float32,
                                shape=[dense_size, class_num])

            b2 = tf.get_variable("b", shape=[class_num],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            pred = tf.matmul(output, W2) + b2
        self.logits = tf.reshape(pred, [-1, num_step,class_num])
        self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(
            self.logits, self.trans_params, self.mask)

        if is_test:
            return
        loss_weight = [[1] * 13]
        loss_weight[0][0] = 0.1
        with tf.variable_scope('crf_loss'):
            self.log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.target, self.mask)

            self.log_likelihood = -self.log_likelihood#tf.nn.relu(-self.log_likelihood)
            self.loss = tf.reduce_mean(self.log_likelihood)
        loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary = loss_summary#tf.summary.merge([loss_summary, accuracy_summary])#
        if is_training:
            return
        self.lr = tf.Variable(0.0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.globle_step = tf.Variable(0, name="globle_step", trainable=False)


        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)



    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self.new_lr: lr_value})

    def assign_new_batch_size(self, session, batch_size_value):
        session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})

    @staticmethod
    def predict(logits,mask,trans_params):
        viterbi_seqs = []
        viterbi_scores =[]
        for id,logit in enumerate(logits):#, self.mask)
            logit = logit[:mask[id]]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_seqs.append(viterbi_seq)
            viterbi_scores.append(viterbi_score)
        return viterbi_seqs,viterbi_scores

