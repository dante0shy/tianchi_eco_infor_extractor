from lstm_crf .data_loader import Data_loader
import os
import time
# from lstm_way.rnn_model import RNN_Model
from lstm_crf.lstm_crf_model import LSTM_CRF
from lstm_crf.tf_config import Config
import tensorflow as tf
import pickle as pkl
import pprint
import numpy as np


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def rebuild(l_s):
    tmp = ''
    for x in l_s:
        if tmp:
            if type(x) == float or type(x) ==np.float64:
                tmp = tmp + str(x)[0:6] + '\t' + ('\t' if len(str(x)[0:6]) < 4 else '')
            else:
                tmp =  tmp + str(x)+ '\t'+('\t' if len(str(x))<4 else '')
        else:
            if type(x) == float or type(x) ==np.float64:
                tmp = str(x)[0:6]+ '\t' + ('\t' if len(str(x)[0:6]) < 4 else '')
            else:
                tmp = str(x)+ '\t'+('\t' if len(str(x))<4 else '')
    # tmp = '\t\t'.join(tmp)
    return tmp

def evaluate(sess,val_model,trans_params , data, global_step=None,val_summary = None):
    correct_num = 0
    total_num = 0
    total_loss = 0
    c=0
    t=0
    total_num_x = 0
    metrix= np.zeros([13,13]).astype(np.int32)
    num = 0
    for step, (x, y, mask_x) in enumerate(data.load_data_batch(batch_size=Config.batch_size)):

        fetches = [val_model.logits,val_model.trans_params,val_model.loss,val_model.viterbi_sequence,val_model.viterbi_score]
        # fetches = val_model.correct_num
        feed_dict = {}
        feed_dict[val_model.input_data] = x
        feed_dict[val_model.target] = y
        feed_dict[val_model.mask] = mask_x
        feed_dict[val_model.keep_prob] = 1.
        feed_dict[val_model.is_training] = False
        val_model.assign_new_batch_size(sess, len(x))
        logits,_,loss,seq,score = sess.run(fetches, feed_dict)
        # seq,score = val_model.predict(logits,mask_x,trans_params)

        len_x = 0
        for idx,pre_seq  in enumerate(seq):

            for i, pre in enumerate(pre_seq):
                total_num_x += 1
                len_x += 1
                if mask_x[idx]<i:
                    break
                t+=1
                metrix[ pre,y[idx][i]] += 1
                if pre  == y[idx][i]:
                    c += 1

                if pre not  in [0] or y[idx][i] not  in [0]:
                    total_num += 1
                    if  pre == y[idx][i]:

                        correct_num +=1
                    # else:
                    #     if  y[idx][i] == 1 or pre ==1:
                    #         print(x[idx][i])len(x)*(cost-cost_t)/(len(x)+num)
        total_loss = total_loss + (loss-total_loss)*len(x)/(len(x)+num)
        num += len(x)
    print(float(c)/t)
    metrix=metrix.tolist()
    l = ["o","pc", "dt", "t", "u", "p", "d", "l",  "pt", "py", "loc", "po",'s']
    for idx,cof in enumerate(metrix):

        print('{}\t:\t{}'.format(l[idx],rebuild(cof)))
    pre = np.array([float(x[i]) for i,x in enumerate(metrix)])/(np.sum(metrix,axis=0))
    print('pre\t:\t{}'.format(rebuild(pre)))
    rec = np.array([float(x[i]) for i, x in enumerate(metrix)]) / (np.sum(metrix,axis=1))
    print('rec\t:\t{}'.format(rebuild(rec)))
    accuracy = float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy', accuracy)
    dev_summary = sess.run(dev_summary)
    if val_summary:
        val_summary.add_summary(dev_summary, global_step)
        val_summary.flush()
    return accuracy,total_loss

def run_epoch(sess,model,val_model,data_t,data_v,global_step,train_summary,val_summary):
    num = 0
    accuracy_t = 0
    cost_t = 0
    trans_params = None
    for step, (x,y,mask) in enumerate(data_t.load_data_batch(batch_size=Config.batch_size)):

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.target] = y
        feed_dict[model.mask] = mask
        feed_dict[model.keep_prob] = Config.keep_prob
        feed_dict[model.is_training] = True
        model.assign_new_batch_size(sess, len(x))
        fetches = [model.optimizer,model.loss, model.summary,model.trans_params]#, model.accuracy,model.out
        opt,cost,summary,trans_params = sess.run(fetches, feed_dict)#accuracy,,out
        train_summary.add_summary(summary, global_step)
        train_summary.flush()
        # accuracy_t += len(x)*accuracy

        cost_t += len(x)*(cost-cost_t)/(len(x)+num)
        num+=len(x)
        global_step += 1
    valid_accuracy,val_loss = evaluate(sess, model,trans_params , data_v, global_step, val_summary)
    print("the %i step, train cost is: %f and the valid loss is %f and the valid accuracy is %f" % (
        global_step, cost, val_loss, valid_accuracy))#accuracy
    print("the %i step, train cost_T is: %f and the train accuracy_T is %f " % (
        global_step, cost_t, accuracy_t/num))
    return  global_step,trans_params,valid_accuracy

def train_step():
    print 'load date'
    config = Config()
    eval_config = Config()
    eval_config.keep_prob = 1.0
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    loader_train = Data_loader(config.train_data,config.char_list,config.label_list,word_len=config.max_vector_len)
    loader_val = Data_loader(config.val_data,config.char_list,config.label_list,word_len=config.max_vector_len)
    # loader.load_data()
    with tf.Graph().as_default(),tf.Session(config=tf_config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess,ui_type='readline')

        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        initializer = tf.truncated_normal_initializer(stddev=Config.init_scale)#-1*FLAGS.init_scale,1*FLAGS.init_scale
        # initializer = tf.random_uniform_initializer(-1*FLAGS.init_scale,1*FLAGS.init_scale)
        with tf.variable_scope("model",initializer=initializer):
            model = LSTM_CRF(config=config)

        with tf.variable_scope("model",reuse=True):#,initializer=initializer
            valid_model = LSTM_CRF(config=eval_config,is_training=1)

        train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # dev_summary_op = tf.merge_summary([valid_model.loss_summary,valid_model.accuracy])
        val_summary_dir = os.path.join(eval_config.out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints_v2"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = None
        # ckpt = tf.train.get_checkpoint_state("/home/xinmatrix/TwoTB/PycharmProjects/deja_receipt_ocr_v3/lstm_crf/model/checkpoints_v1")
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        # tf.global_variables_initializer().run()
        else:
            tf.initialize_all_variables().run()

        global_steps = 1
        begin_time = int(time.time())
        val_acc_max = 0
        for i in range(config.num_epoch):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)
            model.assign_new_lr(sess,config.lr*lr_decay)
            global_steps,trans_params,val_acc=run_epoch(sess, model, valid_model, loader_train,loader_val, global_steps, train_summary_writer, val_summary_writer)

            if i% config.checkpoint_every==0 or (val_acc>val_acc_max and val_acc>=0.8):
                path = saver.save(sess,checkpoint_prefix,global_steps)
                pkl.dump(trans_params,open(os.path.join(checkpoint_dir,'trans_parm.pkl'),'w'))
                print("Saved model chechpoint to{}\n".format(path))
                if val_acc>val_acc_max:
                    val_acc_max = val_acc
                # break
            # break
        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        # test_accuracy,val_loss=evaluate(valid_model,trans_params , sess, loader_val)
        # print("the test data accuracy is %f loss in %f"%test_accuracy%val_loss)
        print("program end!")



def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
