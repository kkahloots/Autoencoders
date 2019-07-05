"""
VAE_graph.py:
Tensorflow Graph for the Variational Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://arxiv.org/pdf/1606.05908.pdf"

import tensorflow as tf
import numpy as np
import utils.constants as const
from base.base_graph import BaseGraph
import base.losses as losses

'''
This is the Main VAEGraph.
'''
class VAEGraph(BaseGraph):
    def __init__(self, configuration):
        BaseGraph.__init__(self)
        self.__dict__.update(configuration)
        self.x_flat_dim = self.width * self.height * self.num_channels
        self.build_graph()

    def build_graph(self):
        self.create_inputs()
        self.create_graph()
        self.create_loss_optimizer()
    
    def create_inputs(self):
        with tf.variable_scope('inputs', reuse=self.reuse):
            self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.num_channels], name='x_batch')
            self.x_batch_flat = tf.reshape(self.x_batch , [-1,self.x_flat_dim])
            
            self.latent_batch = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim], name='latent_batch')
            self.lr = tf.placeholder_with_default(self.learning_rate, shape=None, name='lr')

            self.sample_batch = tf.random_normal((self.batch_size, self.latent_dim), -1, 1, dtype=tf.float32)

    ''' 
    ------------------------------------------------------------------------------
                                     GRAPH FUNCTIONS
    ------------------------------------------------------------------------------ 
    '''
    def create_graph(self):
        print('\n[*] Defining encoders...')
        with tf.variable_scope('encoder_mean', reuse=self.reuse):
            Qlatent_x_mean = self.create_encoder(input_=self.x_batch if self.isConv else self.x_batch_flat,
                            hidden_dim=self.hidden_dim,
                            output_dim=self.latent_dim,
                            num_layers=self.num_layers,
                            transfer_fct=self.transfer_fct,
                            act_out=None,
                            reuse=self.reuse,
                            kinit=self.kinit,
                            bias_init=self.bias_init,
                            drop_rate=self.dropout,
                            prefix='enmean_',
                            isConv=self.isConv)

            self.encoder_mean = Qlatent_x_mean.output

        with tf.variable_scope('encoder_var', reuse=self.reuse):
            Qlatent_x_var = self.create_encoder(input_=self.x_batch if self.isConv else self.x_batch_flat,
                                                 hidden_dim=self.hidden_dim,
                                                 output_dim=self.latent_dim,
                                                 num_layers=self.num_layers,
                                                 transfer_fct=self.transfer_fct,
                                                 act_out=tf.nn.softplus,
                                                 reuse=self.reuse,
                                                 kinit=self.kinit,
                                                 bias_init=self.bias_init,
                                                 drop_rate=self.dropout,
                                                 prefix='envar_',
                                                 isConv=self.isConv)

            self.encoder_var = Qlatent_x_var.output

        print('\n[*] Reparameterization trick...')
        self.encoder_logvar = tf.log(self.encoder_var + const.epsilon)
        eps = tf.random_normal((self.batch_size, self.latent_dim), 0, 1, dtype=tf.float32)
        self.latent = tf.add(self.encoder_mean, tf.multiply(tf.sqrt(self.encoder_var), eps))

        self.latent_batch = self.latent

        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder_mean', reuse=self.reuse):
            Px_latent_mean = self.create_decoder(input_=self.latent_batch,
                                            hidden_dim=self.hidden_dim,
                                            output_dim=self.x_flat_dim,
                                            num_layers=self.num_layers,
                                            transfer_fct=self.transfer_fct,
                                            act_out=tf.nn.sigmoid,
                                            reuse=self.reuse,
                                            kinit=self.kinit,
                                            bias_init=self.bias_init,
                                            drop_rate=self.dropout,
                                            prefix='de_',
                                            isConv=self.isConv)

            self.x_recons_flat = Px_latent_mean.output
        self.x_recons = tf.reshape(self.x_recons_flat , [-1,self.width, self.height, self.num_channels])

    '''  
    ------------------------------------------------------------------------------
                                     LOSSES
    ------------------------------------------------------------------------------ 
    '''
    def create_loss_optimizer(self):
        print('[*] Defining Loss Functions and Optimizer...')
        with tf.name_scope('reconstruct'):
            self.reconstruction = losses.get_ell(self.x_batch_flat, self.x_recons_flat)
        self.loss_reconstruction_m = tf.reduce_mean(self.reconstruction)

        with tf.variable_scope('L2_loss', reuse=self.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope('ae_loss', reuse=self.reuse):
            self.ae_loss = tf.add(tf.reduce_mean(self.reconstruction), self.l2*self.L2_loss, name='ae_loss')

        with tf.variable_scope('kl_loss', reuse=self.reuse):
            self.kl_loss = losses.get_kl(self.encoder_mean, self.encoder_logvar)
        self.kl_loss_m = tf.reduce_mean(self.kl_loss)

        with tf.variable_scope('vae_loss', reuse=self.reuse):
            self.vae_loss = tf.add(self.ae_loss, self.kl_loss_m)

        with tf.variable_scope('optimizer', reuse=self.reuse):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.vae_loss, global_step=self.global_step_tensor)

        self.losses = ['VAE', 'AE', 'reconstruction', 'L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, aeloss, recons, L2_loss = session.run(tensors, feed_dict=feed_dict)
        return loss, aeloss, recons, L2_loss
    
    def evaluate_epoch(self, session, x):
        tensors = [self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, aeloss, recons, L2_loss
