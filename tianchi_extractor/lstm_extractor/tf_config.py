# -*- coding: UTF-8 -*-

import tensorflow as tf
import os
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
label_dict = [ 'z', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 't',]
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size',64,'the batch_size of the training procedure')
flags.DEFINE_float('lr',0.01,'the learning rate')
flags.DEFINE_float('lr_decay',0.89,'the learning rate decay')
flags.DEFINE_integer('hidden_neural_size',256,'LSTM hidden neural size')
flags.DEFINE_integer('embed_dim',256,'embed_dim')
flags.DEFINE_integer('hidden_layer_num',1,'LSTM hidden layer num')
flags.DEFINE_string('dataset_path','train.json','dataset path')
# flags.DEFINE_string('dataset_path','word_data.pkl','dataset path')
flags.DEFINE_integer('max_len',1000,'max_len of training sentence')
flags.DEFINE_integer('valid_num',2,'epoch num of validation')
flags.DEFINE_integer('checkpoint_num',10,'epoch num of checkpoint')
flags.DEFINE_float('init_scale',0.1,'init scale')
flags.DEFINE_integer('class_num',len(label_dict),'class num')
flags.DEFINE_float('keep_prob',0.5,'dropout rate')
flags.DEFINE_integer('num_epoch',50000,'num epoch')
flags.DEFINE_integer('max_decay_epoch',10,'num epoch')
flags.DEFINE_integer('max_grad_norm',5,'max_grad_norm')
flags.DEFINE_string('out_dir',os.path.join(base_path,'extras','model'),'output directory')
flags.DEFINE_integer('check_point_every',2,'checkpoint every num epoch ')
flags.DEFINE_integer('max_vector_len',1000,'max vector len ')
flags.DEFINE_integer('vector_len',2,'vector len ')#107
flags.DEFINE_integer('word_len',171785,'vector len ')
flags.DEFINE_integer('seq_class',3,'seq_class')
flags.DEFINE_string('train_data',os.path.join(base_path,'extras','train.json'),'train data')
flags.DEFINE_string('val_data',os.path.join(base_path,'extras','val.json'),'val data')
flags.DEFINE_string('char_list',os.path.join(base_path,'extras','word_list_0.json'),'char list')

class Config(object):
    hidden_neural_size = FLAGS.hidden_neural_size
    hidden_layer_num = FLAGS.hidden_layer_num
    class_num = FLAGS.class_num
    keep_prob = FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm = FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    valid_num = FLAGS.valid_num
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every
    max_vector_len = FLAGS.max_vector_len
    vector_len = FLAGS.vector_len
    init_scale = FLAGS.init_scale
    train_data = FLAGS.train_data
    val_data = FLAGS.val_data
    char_list = FLAGS.char_list
    word_len = FLAGS.word_len
    embed_dim = FLAGS.embed_dim
    seq_class = FLAGS.seq_class

path_format = os.path.join(base_path,'extras','data_out','{}','par','{}.json')
