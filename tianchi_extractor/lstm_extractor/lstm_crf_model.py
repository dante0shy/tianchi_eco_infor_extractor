

import tensorflow as tf

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

import numpy as np
class LSTM_CRF():
    def __init__(self,config,is_training= 0,is_test = 0):
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
        self.batch_size = tf.Variable(config.batch_size, dtype=tf.int32, trainable=False)
        num_step = config.num_step
        class_num = config.class_num
        words_len = config.word_len
        embed_dim = config.embed_dim
        self.input_data = tf.placeholder(tf.int32, [None, num_step])
        self.target = tf.placeholder(tf.int32, [None,num_step])
        self.mask = tf.placeholder(tf.int32, [None,])
        self.seq_class = tf.placeholder(tf.float32, [None,num_step,config.seq_class])
        hidden_neural_size = config.hidden_neural_size
        hidden_layer_num = config.hidden_layer_num
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self.is_training =  tf.placeholder(tf.bool)
        trans_params = tf.get_variable("transitions",[class_num,class_num],tf.float32)
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        input_data = tf.reshape(self.input_data, [-1, config.vector_len])
        self.pad = tf.placeholder(tf.float32, [None, 1, embed_dim, 1], name='pad')
        l2_loss = tf.constant(0.01)
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding",[words_len,embed_dim],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            # emb = tf.expand_dims(inputs, -1)
        # seq_class = tf.reshape(self.seq_class,[-1,num_step,3,1])
        input_emded = tf.concat([inputs,self.seq_class],axis=2)
        input_data = tf.reshape(input_emded, [-1, embed_dim+3])
        with tf.variable_scope("dense0"):
            with tf.variable_scope("dense0_1"):
                W0_1 = tf.get_variable("W", dtype=tf.float32,
                                     shape=[embed_dim+3, 256])
                input_data = tf.matmul(input_data, W0_1)
                input_data = tf.contrib.layers.batch_norm(input_data,is_training =self.is_training)
                input_data = tf.nn.tanh(input_data)
        input_data = tf.reshape(input_data,[-1, num_step,1,256])
        # input_data = tf.reshape(input_data,[-1,num_step])

        pooled_concat = []
        # reduced = np.int32(np.ceil((config.max_vector_len) * 1.0 / 4))
        for i, filter_size in enumerate([1,3,5,7]):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                filter_shape = [filter_size, 1, 256, 128]
                conv =self.conv_2d_relu(input_data,filter_shape=filter_shape,stride_size=[1, 1, 1, 1])
                conv = tf.reshape(conv, [-1, config.num_step, 128])
                pooled_concat.append(conv)

        pooled_concat = tf.concat(pooled_concat, 2)

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
        self.logits = tf.reshape(pred, [self.batch_size, num_step,class_num])
        # mask = tf.reshape(self.mask,[-1,1])
        self.out = self.logits
        self.viterbi_sequence, self.viterbi_score =self.crf_decode(self.logits, trans_params, self.mask)

        if is_test:
            return
        with tf.variable_scope('crf_loss'):
            self.log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.target, self.mask)

            self.log_likelihood = -self.log_likelihood#tf.nn.relu(-self.log_likelihood)
            self.loss = tf.reduce_mean(self.log_likelihood)
        loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary = loss_summary#tf.summary.merge([loss_summary, accuracy_summary])#
        self.trans_params =trans_params

        if is_training:
            return
        self.lr = tf.Variable(0.0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.globle_step = tf.Variable(0, name="globle_step", trainable=False)


        self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

    def conv_2d_relu(self,input,filter_shape,stride_size, padding = 'SAME'):
        W_conv_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
        # b_conv_1 = tf.Variable(tf.constant(0.1, shape=[filter_shape[-1]]), name='b')
        # padded_input = tf.pad(input_image, padding, "CONSTANT")
        conv = tf.nn.conv2d(input, W_conv_1, strides=stride_size, padding=padding)
        output = tf.contrib.layers.batch_norm(conv,is_training =self.is_training)
        output = tf.nn.relu(output, name='relu')
        return output

    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self.new_lr: lr_value})

    def assign_new_batch_size(self, session, batch_size_value):
        session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})

    def crf_decode(self,potentials, transition_params, sequence_length):
        """Decode the highest scoring sequence of tags in TensorFlow.

        This is a function for tensor.

        Args:
          potentials: A [batch_size, max_seq_len, num_tags] tensor, matrix of
                    unary potentials.
          transition_params: A [num_tags, num_tags] tensor, matrix of
                    binary potentials.
          sequence_length: A [batch_size] tensor, containing sequence lengths.

        Returns:
          decode_tags: A [batch_size, max_seq_len] tensor, with dtype tf.int32.
                      Contains the highest scoring tag indicies.
          best_score: A [batch_size] tensor, containing the score of decode_tags.
        """
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        num_tags = potentials.get_shape()[2].value

        # Computes forward decoding. Get last score and backpointers.
        crf_fwd_cell = tf.contrib.crf.CrfDecodeForwardRnnCell(transition_params)
        initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
        inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
        sequence_length_less_one = math_ops.maximum(0, sequence_length - 1)
        backpointers, last_score = rnn.dynamic_rnn(
            crf_fwd_cell,
            inputs=inputs,
            sequence_length=sequence_length_less_one ,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)  # [B, T - 1, O], [B, O]
        backpointers = gen_array_ops.reverse_sequence(
            backpointers, sequence_length_less_one , seq_dim=1)  # [B, T-1, O]

        # Computes backward decoding. Extract tag indices from backpointers.
        crf_bwd_cell = tf.contrib.crf.CrfDecodeBackwardRnnCell(num_tags)
        initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),
                                      dtype=dtypes.int32)  # [B]
        initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
        decode_tags, _ = rnn.dynamic_rnn(
            crf_bwd_cell,
            inputs=backpointers,
            sequence_length=sequence_length_less_one ,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)  # [B, T - 1, 1]
        decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
        decode_tags = array_ops.concat([initial_state, decode_tags], axis=1)  # [B, T]
        decode_tags = gen_array_ops.reverse_sequence(
            decode_tags, sequence_length_less_one, seq_dim=1)  # [B, T]

        best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
        return decode_tags, best_score

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
