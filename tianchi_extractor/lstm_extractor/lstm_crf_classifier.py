from  tianchi_extractor.lstm_extractor.text_process import Data_loader
import os
import time
# from lstm_way.rnn_model import RNN_Model
from tianchi_extractor.lstm_extractor.lstm_crf_model import LSTM_CRF
from tianchi_extractor.lstm_extractor .tf_config import Config
import tensorflow as tf
import pickle as pkl
import glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Classifier():

    def __init__(self):
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.config = Config()
        self.config.keep_prob = 1.
        # self.config.batch_size = 1
        self.model_path = os.path.abspath(
            os.path.join(self.config.out_dir, "checkpoints_v1"))
        self.sess = tf.Session(config=self.tf_config)
        # with   as sess:
        with tf.variable_scope("model"):
            self.model = LSTM_CRF(config=self.config,is_test=1)
        self.sess.run(tf.initialize_all_variables())

        if self.model_path:
            print("loading model from {}".format(self.model_path))
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.sess, '/home/xinmatrix/TwoTB/tianchi/extras/model/checkpoints_v1/model-5698')
        else:
            raise ('no model')

    def run_app(self,sess,data):
        total_pre = []
        for x,d,mask in data.load_data_batch(64):
            fetches = [self.model.logits,self.model.viterbi_sequence,self.model.viterbi_score]
            feed_dict = {}
            feed_dict[self.model.input_data] = x
            feed_dict[self.model.batch_size] = len(x)
            feed_dict[self.model.target] = [[0]*1000]*len(x)
            feed_dict[self.model.mask] = mask
            feed_dict[self.model.seq_class] = d
            feed_dict[self.model.keep_prob] = 1.
            feed_dict[self.model.is_training] = False
            print('mask : {}'.format(len(x)))
            logits,prediction, confidence= sess.run(fetches, feed_dict)
            pre = [p[:mask[idx]] for idx ,p in  enumerate(prediction.tolist())]
            for p in pre:
                total_pre.extend(p)
        # print(t)
        # seq, score = self.model.predict(logits, mask, self.trans_params)
        return  total_pre


    def predict(self,data):
        print 'load date'
        begin_time = (time.time())
        prediction = self.run_app(self.sess,data)
        print("the test is finished")
        end_time=(time.time())
        print("takes %f seconds\n"%(end_time-begin_time))
        return  prediction


if __name__ == "__main__":
    classifier = Classifier()
    base_dir = '/home/xinmatrix/TwoTB/databack/Picture_Receipt/*/*'
    # base_dir = '/home/xinmatrix/TwoTB/databack/brands_reodered_for_train/*/*'
    files = glob.glob(base_dir)
    ocrhandler = OCRHandler()
    t = 0.
    f= 0.
    for file in files:
        label = int(file.split('/')[-2])
        try:
            text_response, cv2_img, ocr_angle, real_angle, shape, shape_new = ocrhandler.detect_ocr_text(file,None)
            detail_list, mask_list = classifier.processer.build_data_lsit([text_response])
        except:
            continue
        pre, score = classifier.predict( detail_list, mask_list)
        pre_label = classifier.processer.brand_label[pre[0]]
        print('{}:{}'.format(label,pre_label))
        if label==pre_label:
            t = t+1
        else:
            f = f+1

        print("{} : pre {} ground {}".format(file,pre_label,label))
    print(t/(t+f))
