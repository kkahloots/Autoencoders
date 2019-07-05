

import tensorflow as tf
from networks.dense_net import DenseNet
from networks.conv_net import ConvNet3
from networks.deconv_net import DeconvNet3

import utils.constants as const

class BaseGraph:
    def __init__(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)
        
    def build_graph(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def evaluate_epoch(self):
        raise NotImplementedError

    '''  
    ------------------------------------------------------------------------------
                                     GENERATE LATENT and RECONSTRUCT
    ------------------------------------------------------------------------------ 
    '''

    def reconst_loss(self, session, x):
        tensors = [self.reconstruction]
        feed = {self.x_batch: x}
        return session.run(tensors, feed_dict=feed)

    def decay_lr(self, session):
        self.lr = tf.multiply(0.1, self.lr)
        nlr = session.run(self.lr)

        if nlr > const.min_lr:
            print('decaying learning rate ... ')

            tensors = [self.lr]
            feed_dict = {self.lr: nlr}
            nlr = session.run(tensors, feed_dict=feed_dict)[0]
            nlr = session.run(self.lr)
            nlr = round(nlr, 8)
            print('new learning rate: {}'.format(nlr))

    '''  
    ------------------------------------------------------------------------------
                                         GRAPH OPERATIONS
    ------------------------------------------------------------------------------ 
    '''

    def encode(self, session, inputs):
        tensors = [self.latent]
        feed_dict = {self.x_batch: inputs}
        return session.run(tensors, feed_dict=feed_dict)

    def decode(self, session, latent):
        tensors = [self.x_recons]
        feed_dict = {self.latent: latent}
        return session.run(tensors, feed_dict=feed_dict)

    def sample(self, session):
        random_latent = session.run(tf.random_normal((self.batch_size, self.latent_dim), \
                                                     self.latent_min + (2*self.latent_std),
                                                     self.latent_max + (2*self.latent_std), \
                                                     dtype=tf.float32))
        tensors = [self.x_recons]
        feed_dict = {self.latent_batch: random_latent}
        return session.run(tensors, feed_dict=feed_dict)[0]


    '''  
    ------------------------------------------------------------------------------
                                     ENCODER-DECODER
    ------------------------------------------------------------------------------ 
    '''

    def create_encoder(self, input_, hidden_dim, output_dim, num_layers, transfer_fct, \
                       act_out, reuse, kinit, bias_init, drop_rate, prefix, isConv):
        latent_ = DenseNet(input_=input_,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           transfer_fct=transfer_fct,
                           act_out=act_out,
                           reuse=reuse,
                           kinit=kinit,
                           bias_init=bias_init,
                           drop_rate=drop_rate,
                           prefix=prefix) if not isConv else \
            ConvNet3(input_=input_,
                     hidden_dim=hidden_dim,
                     output_dim=output_dim,
                     num_layers=num_layers,
                     transfer_fct=transfer_fct,
                     act_out=act_out,
                     reuse=reuse,
                     kinit=kinit,
                     bias_init=bias_init,
                     drop_rate=drop_rate,
                     prefix=prefix)
        return latent_

    def create_decoder(self, input_, hidden_dim, output_dim, num_layers, transfer_fct, \
                       act_out, reuse, kinit, bias_init, drop_rate, prefix, isConv):
        recons_ = DenseNet(input_=input_,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           num_layers=num_layers,
                           transfer_fct=transfer_fct,
                           act_out=act_out,
                           reuse=reuse,
                           kinit=kinit,
                           bias_init=bias_init,
                           drop_rate=drop_rate,
                           prefix=prefix) if not isConv else \
            DeconvNet3(input_=input_,
                       num_layers=num_layers,
                       hidden_dim=hidden_dim,
                       output_dim=output_dim,
                       width=self.width,
                       height=self.height,
                       nchannels=self.num_channels,
                       transfer_fct=transfer_fct,
                       act_out=act_out,
                       reuse=reuse,
                       kinit=kinit,
                       bias_init=bias_init,
                       drop_rate=drop_rate,
                       prefix=prefix)
        return recons_



