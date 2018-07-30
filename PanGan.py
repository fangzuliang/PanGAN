import tensorflow as tf
import numpy as np

class PanGan(object):
    
    def __init__(self, pan_size, ms_size, batch_size,num_spectrum, ratio,init_lr=0.001,lr_decay_rate=0.99,lr_decay_step=1000, is_training=True):
        
        self.num_spectrum=num_spectrum
        self.is_training=is_training
        self.ratio = ratio
        self.batch_size=batch_size
        self.pan_size=pan_size
        self.ms_size=ms_size
        self.init_lr=init_lr
        self.lr_decay_rate=lr_decay_rate
        self.lr_decay_step=lr_decay_step
        self.build_model(pan_size, ms_size, batch_size,num_spectrum, is_training)
        
    def build_model(self, pan_size, ms_size, batch_size, num_spectrum, is_training):
        with tf.name_scope('input'):
            self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1), name='pan_placeholder')
            self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,ms_size, ms_size, num_spectrum), name='ms_placeholder')
            self.pan_img_hp=self.high_pass(self.pan_img, 'pan')
            self.ms_img_hp=self.high_pass(self.ms_img, 'ms')
        if is_training:
            with tf.name_scope('PanSharpening'):
                self.PanSharpening_img= self.PanSharpening_model(self.pan_img, self.ms_img)
                self.PanSharpening_img_=tf.image.resize_images(images=self.PanSharpening_img, size=[ms_size, ms_size],
                                                               method=tf.image.ResizeMethod.BILINEAR)
                self.PanSharpening_img_hp=self.high_pass(self.PanSharpening_img)
            
            with tf.name_scope('d_loss'):
                with tf.name_scope('spatial_loss'):
                    spatial_pos=self.spatial_discriminator(self.pan_img, reuse=False)
                    spatial_neg=self.spatial_discriminator(self.PanSharpening_img, reuse=True)
                    spatial_pos_loss= tf.reduce_mean(tf.square(spatial_pos-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                    spatial_neg_loss= tf.reduce_mean(tf.square(spatial_neg-tf.zeros(shape=[batch_size,1], dtype=tf.float32)))
                    self.spatial_loss=spatial_pos_loss + spatial_neg_loss
                    tf.summary.scalar('spatial_loss', self.spatial_loss)
                with tf.name_scope('spectrum_loss'):
                    spectrum_pos=self.spectrum_discriminator(self.ms_img, reuse=False)
                    spectrum_neg=self.spectrum_discriminator(self.PanSharpening_img_, reuse=True)
                    spectrum_pos_loss= tf.reduce_mean(tf.square(spectrum_pos-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                    spectrum_neg_loss= tf.reduce_mean(tf.square(spectrum_neg-tf.zeros(shape=[batch_size,1], dtype=tf.float32)))
                    self.spectrum_loss=spectrum_pos_loss + spectrum_neg_loss
                    tf.summary.scalar('spectrum_loss', self.spectrum_loss)
            
            with tf.name_scope('g_loss'):
                spatial_loss_ad= tf.reduce_mean(tf.square(spatial_neg-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                tf.summary.scalar('spatial_loss_ad', spatial_loss_ad)
                spectrum_loss_ad=tf.reduce_mean(tf.square(spectrum_neg-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                tf.summary.scalar('spectrum_loss_ad', spectrum_loss_ad)
                g_spatital_loss= tf.reduce_mean(tf.square(self.PanSharpening_img_hp-self.pan_img_hp))
                tf.summary.scalar('g_spatital_loss', g_spatital_loss)
                g_spectrum_loss=tf.reduce_mean(tf.square(self.PanSharpening_img_-self.ms_img))
                tf.summary.scalar('g_spectrum_loss', g_spectrum_loss)
                self.g_loss= spatial_loss_ad + spectrum_loss_ad + 7*g_spatital_loss + g_spectrum_loss
                #self.g_loss=g_spatital_loss + g_spectrum_loss
                tf.summary.scalar('g_loss', self.g_loss)
        else:
            self.PanSharpening_img=self.PanSharpening_model(self.pan_img, self.ms_img)

    def train(self):
        t_vars = tf.trainable_variables()
        d_spatial_vars = [var for var in t_vars if 'spatial_discriminator' in var.name]
        d_spectrum_vars=[var for var in t_vars if 'spectrum_discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Pan_model' in var.name]
        with tf.name_scope('train_step'):
            self.global_step=tf.contrib.framework.get_or_create_global_step()
            self.learning_rate=tf.train.exponential_decay(self.init_lr, global_step=self.global_step, decay_rate=self.lr_decay_rate,
                                                          decay_steps=self.lr_decay_step)
            tf.summary.scalar('global learning rate', self.learning_rate)
            self.train_Pan_model=tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars, global_step=self.global_step)
            self.train_spatial_discrim=tf.train.AdamOptimizer(self.learning_rate).minimize(self.spatial_loss, var_list=d_spatial_vars)
            self.train_spectrum_discrim=tf.train.AdamOptimizer(self.learning_rate).minimize(self.spectrum_loss, var_list=d_spectrum_vars)


    def PanSharpening_model(self,pan_img, ms_img):
        with tf.variable_scope('Pan_model'):
            with tf.name_scope('upscale'):
                # de_weight=tf.get_variable('de_weight', [3,3,self.num_spectrum, self.num_spectrum],
                                        # initializer=tf.truncated_normal_initializer(stddev=1e-3) )
                # ms_scale4 = tf.nn.conv2d_transpose(ms_img, de_weight, output_shape=[self.batch_size,self.pan_size,self.pan_size,self.num_spectrum],
                                                   # strides=[1,4,4,1],padding='SAME' )                            
                ms_scale4=tf.image.resize_images(ms_img, [self.pan_size, self.pan_size], method=2)
            input=tf.concat([pan_img, ms_scale4],axis=-1)
            with tf.variable_scope('layer1'):
                weights = tf.get_variable("w1", [9, 9, self.num_spectrum + 1, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [64], initializer=tf.constant_initializer(0.0))
                conv1 = tf.contrib.layers.batch_norm(tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv1= self.lrelu(conv1)
            with tf.variable_scope('layer2'):
                weights = tf.get_variable("w2", [5, 5, 64, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
                conv2 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv2= self.lrelu(conv2)
            with tf.variable_scope('layer3'):
                weights = tf.get_variable("w3", [5, 5, 32 , self.num_spectrum],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b3", [self.num_spectrum], initializer=tf.constant_initializer(0.0))
                conv3 = tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                                     decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)
                conv3= self.lrelu(conv3)

        return conv3

    def spatial_discriminator(self,img_hp,reuse=False):
        with tf.variable_scope('spatial_discriminator', reuse=reuse):
            with tf.variable_scope('layer_1'):
                weights = tf.get_variable("w_1", [3, 3, self.num_spectrum, 16],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_1", [16], initializer=tf.constant_initializer(0.0))
                conv1_spatial = tf.nn.conv2d(img_hp, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
                conv1_spatial = self.lrelu(conv1_spatial)
                # print(conv1_vi.shape)
            with tf.variable_scope('layer_2'):
                weights = tf.get_variable("w_2", [3, 3, 16, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_2", [32], initializer=tf.constant_initializer(0.0))
                conv2_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv1_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv2_spatial = self.lrelu(conv2_spatial)
                # print(conv2_vi.shape)
            with tf.variable_scope('layer_3'):
                weights = tf.get_variable("w_3", [3, 3, 32, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_3", [64], initializer=tf.constant_initializer(0.0))
                conv3_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv2_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv3_spatial = self.lrelu(conv3_spatial)
                # print(conv3_vi.shape)
            with tf.variable_scope('layer_4'):
                weights = tf.get_variable("w_4", [3, 3, 64, 128],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_4", [128], initializer=tf.constant_initializer(0.0))
                conv4_spatial = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv3_spatial, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv4_spatial = self.lrelu(conv4_spatial)
                conv4_spatial = tf.reshape(conv4_spatial, [self.batch_size, 4 * 4 * 128])
            with tf.variable_scope('line_5'):
                weights = tf.get_variable("w_5", [4 * 4 * 128, 1],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b_5", [1], initializer=tf.constant_initializer(0.0))
                line5_spatial = tf.matmul(conv4_spatial, weights) + bias
                # conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return line5_spatial

    def spectrum_discriminator(self,img,reuse=False):
        with tf.variable_scope('spectrum_discriminator', reuse=reuse):
            with tf.variable_scope('layer1_spectrum'):
                weights = tf.get_variable("w1_spectrum", [3, 3, self.num_spectrum, 16],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b1_spectrum", [16], initializer=tf.constant_initializer(0.0))
                conv1_spectrum = tf.nn.conv2d(img, weights, strides=[1, 2, 2, 1], padding='SAME') + bias
                conv1_spectrum = self.lrelu(conv1_spectrum)
                # print(conv1_vi.shape)
            with tf.variable_scope('layer2_spectrum'):
                weights = tf.get_variable("w2_spectrum", [3, 3, 16, 32],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b2_spectrum", [32], initializer=tf.constant_initializer(0.0))
                conv2_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv1_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv2_spectrum = self.lrelu(conv2_spectrum)
                # print(conv2_vi.shape)
            with tf.variable_scope('layer3_spectrum'):
                weights = tf.get_variable("w3_spectrum", [3, 3, 32, 64],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b3_spectrum", [64], initializer=tf.constant_initializer(0.0))
                conv3_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv2_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv3_spectrum = self.lrelu(conv3_spectrum)
                # print(conv3_vi.shape)
            with tf.variable_scope('layer4_spectrum'):
                weights = tf.get_variable("w4_spectrum", [3, 3, 64, 128],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b4_spectrum", [128], initializer=tf.constant_initializer(0.0))
                conv4_spectrum = tf.contrib.layers.batch_norm(
                    tf.nn.conv2d(conv3_spectrum, weights, strides=[1, 2, 2, 1], padding='SAME') + bias, decay=0.9,
                    updates_collections=None, epsilon=1e-5, scale=True)
                conv4_spectrum = self.lrelu(conv4_spectrum)
                conv4_spectrum = tf.reshape(conv4_spectrum, [self.batch_size, 1 * 1 * 128])
            with tf.variable_scope('line5_spectrum'):
                weights = tf.get_variable("w5_spectrum", [1 * 1 * 128, 1],
                                          initializer=tf.truncated_normal_initializer(stddev=1e-3))
                bias = tf.get_variable("b5_spectrum", [1], initializer=tf.constant_initializer(0.0))
                line5_spectrum = tf.matmul(conv4_spectrum, weights) + bias
                # conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return line5_spectrum
        
    def high_pass(self, img, type='PanSharepening'):
        if type=='pan':
            input=img
            for i in range(self.num_spectrum-1):
                input=tf.concat([input,img],axis=-1)
            img=input
        if type == 'ms':
            img=tf.image.resize_images(img, [self.pan_size, self.pan_size], method=2)
        blur_kerel=np.zeros(shape=(9,9,self.num_spectrum, self.num_spectrum), dtype=np.float32)
        value=1/81*np.ones(shape=(9,9), dtype=np.float32)
        for i in range(self.num_spectrum):
            blur_kerel[:,:,i,i]=value
        img_lp=tf.nn.conv2d(img,tf.convert_to_tensor(blur_kerel),strides=[1,1,1,1], padding='SAME')
        img_hp=img-img_lp
        return tf.abs(img_hp)
    def high_pass_1(self, img, type='ms'):
        if type=='pan':
            input=img
            for i in range(self.num_spectrum-1):
                input=tf.concat([input,img],axis=-1)
            img=input
        blur_kerel=np.zeros(shape=(3,3,self.num_spectrum, self.num_spectrum), dtype=np.float32)
        value=np.array([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])
        for i in range(self.num_spectrum):
            blur_kerel[:,:,i,i]=value
        img_hp=tf.nn.conv2d(img,tf.convert_to_tensor(blur_kerel),strides=[1,1,1,1], padding='SAME')
        #img_hp=img-img_lp
        return tf.abs(img_hp)
       
    def lrelu(self,x, leak=0.2):
        return tf.maximum(x, leak * x)
        

