import tensorflow as tf

from lstm_crf.tf_config import Config

class LSTM_CRF():
    def __init__(self,config,is_training= 0,is_test = 0):
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
        self.batch_size = tf.Variable(config.batch_size, dtype=tf.int32, trainable=False)
        num_step = config.num_step
        class_num = config.class_num
        words_len = config.word_len
        self.input_data = tf.placeholder(tf.float32, [None, num_step,config.vector_len])
        self.target = tf.placeholder(tf.int32, [None,num_step])
        self.mask = tf.placeholder(tf.int32, [None,])
        hidden_neural_size = config.hidden_neural_size
        hidden_layer_num = config.hidden_layer_num
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self.is_training =  tf.placeholder(tf.bool)
        self.trans_params = tf.get_variable("transitions",[class_num,class_num],tf.float32)
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        l2_loss = tf.constant(0.01)
        input_data = tf.reshape(self.input_data, [-1, config.vector_len])
        with tf.variable_scope("dense0_1"):
            W0_1 = tf.get_variable("W", dtype=tf.float32,
                                 shape=[config.vector_len, 512])
            # b0_1 = tf.get_variable("b", shape=[512],
            #                      dtype=tf.float32, initializer=tf.zeros_initializer())

            input_data = tf.matmul(input_data, W0_1) #+ b0_1
            input_data = tf.contrib.layers.batch_norm(input_data,is_training = self.is_training)
            input_data = tf.nn.tanh(input_data)
        with tf.variable_scope("dense0_2"):
            W0_2 = tf.get_variable("W", dtype=tf.float32,
                                 shape=[512, 512])
            # b0_2 = tf.get_variable("b", shape=[512],
            #                      dtype=tf.float32, initializer=tf.zeros_initializer())
            input_data = tf.matmul(input_data, W0_2)# + b0_2
            input_data = tf.contrib.layers.batch_norm(input_data,is_training = self.is_training)
            input_data = tf.nn.tanh(input_data)
        input_data = tf.reshape(input_data,[-1, num_step,512])
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_neural_size, forget_bias=1.0)
                                               for _ in range(hidden_layer_num)] , state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_neural_size, forget_bias=1.0)
                                               for _ in range(hidden_layer_num)], state_is_tuple=True)
        _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=input_data,
            dtype=tf.float32,
            sequence_length=self.mask)

        (output_fw, output_bw), _ = _output
        output = tf.concat([output_fw, output_bw],2)


        output = tf.reshape(output,[-1,2 * hidden_neural_size])
        # output = tf.contrib.layers.batch_norm(output)
        # output = tf.nn.sigmoid(output)
        dense_size = 1024
        with tf.variable_scope("dense1"):
            W1 = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * hidden_neural_size, dense_size])

            b1 = tf.get_variable("b", shape=[dense_size],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            output = tf.matmul(output, W1) + b1
            # output = tf.contrib.layers.batch_norm(output)
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

