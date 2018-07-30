import tensorflow as tf
import os 
import numpy as np
import cv2
from PanGan import PanGan

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pan_size',
                           default_value=512,
                           docstring='pan image size')
tf.app.flags.DEFINE_string('ms_size',
                           default_value=128,
                           docstring='ms image size')
tf.app.flags.DEFINE_string('batch_size',
                           default_value=1,
                           docstring='img batch')
tf.app.flags.DEFINE_string('num_spectrum',
                           default_value=1,
                           docstring='spectrum num')
tf.app.flags.DEFINE_string('ratio',
                           default_value=4,
                           docstring='pan image/ms img')
tf.app.flags.DEFINE_string('model_path',
                           default_value='./model/Generator-50000',
                           docstring='pan image/ms img') 
tf.app.flags.DEFINE_string('test_path',
                           default_value='./data/test',
                           docstring='test img data')                            
tf.app.flags.DEFINE_string('result_path',
                           default_value='./result',
                           docstring='result img')                          
tf.app.flags.DEFINE_string('norm',
                           default_value=True,
                           docstring='if norm') 


                           
def main(argv):
    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    model=PanGan(FLAGS.pan_size,FLAGS.ms_size, FLAGS.batch_size, FLAGS.num_spectrum, FLAGS.ratio,0.001, 0.99, 1000,False)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model_path)
        ms_test_path= FLAGS.test_path + '/ir'
        pan_test_path=FLAGS.test_path + '/vi'
        for img_name in os.listdir(ms_test_path):
            pan, ms = read_img(pan_test_path, ms_test_path, img_name,FLAGS)
            PanSharpening= sess.run(model.PanSharpening_img, feed_dict={model.pan_img:pan, model.ms_img:ms})
            PanSharpening=(PanSharpening+0.5)*255.0
            PanSharpening=PanSharpening.squeeze()
            save_path=os.path.join(FLAGS.result_path, img_name)
            cv2.imwrite(save_path, PanSharpening)
            print(img_name + ' done')
            
def read_img(pan_test_path, ms_test_path, img_name, FLAGS):
    pan=np.zeros(shape=(1, FLAGS.pan_size, FLAGS.pan_size, 1))
    ms=np.zeros(shape=(1, FLAGS.ms_size, FLAGS.ms_size, FLAGS.num_spectrum ))
    pan_img_path=os.path.join(pan_test_path, img_name)
    ms_img_path=os.path.join(ms_test_path, img_name)
    pan_img=cv2.imread(pan_img_path, cv2.IMREAD_GRAYSCALE)
    ms_img=cv2.imread(ms_img_path, cv2.IMREAD_GRAYSCALE)
    if FLAGS.norm:
        pan_img=pan_img/255.0-0.5
        ms_img=ms_img/255.0-0.5
    pan[0, :, :, 0]= pan_img
    ms[0, :, :, 0]=ms_img
    return pan, ms
    
if __name__ == '__main__':
    tf.app.run()
    
      
    
