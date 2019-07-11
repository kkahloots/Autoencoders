"""
BayesianAE_graph.py:
Tensorflow Graph for the Bayesian Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://dl.acm.org/citation.cfm?id=3016538"


import tensorflow as tf
import numpy as np
import utils.constants as const
from base.base_graph import BaseGraph
import base.losses as losses

'''
This is the Main BayesianAEGraph.
'''
class BayesianAEGraph(BaseGraph):
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
        print('\n[*] Defining sample from X...')
        input_sample = tf.multiply(tf.ones([self.MC_samples, self.batch_size, self.x_flat_dim]), self.x_batch_flat)
        if self.isConv:
            input_sample = tf.reshape(self.x_batch , [-1, self.width, self.height, self.num_channels])
        print('\n[*] sample shape {}'.format(input_sample.shape))

        print('\n[*] Defining prior ...')
        with tf.variable_scope('prior_mean', reuse=self.reuse):
            Qlatent_x_mean = self.create_encoder(input_=input_sample,
                                                 hidden_dim=self.hidden_dim,
                                                 output_dim=self.latent_dim,
                                                 num_layers=self.num_layers,
                                                 transfer_fct=self.transfer_fct,
                                                 act_out=None,
                                                 reuse=self.reuse,
                                                 kinit=self.kinit,
                                                 bias_init=self.bias_init,
                                                 drop_rate=self.dropout,
                                                 prefix='prior_var_',
                                                 isConv=self.isConv)

            self.prior_mean = Qlatent_x_mean.output

        with tf.variable_scope('prior_var', reuse=self.reuse):
            self.prior_var = tf.nn.sigmoid(self.prior_mean)


        ####################################################################################
        print('\n[*] Defining posterior ...')
        print('\n[*] Defining posterior encoder...')
        with tf.variable_scope('post_encoder', reuse=self.reuse):
            Qlatent_x = self.create_encoder(input_=self.x_batch if self.isConv else self.x_batch_flat,
                                            hidden_dim=self.hidden_dim,
                                            output_dim=self.latent_dim,
                                            num_layers=self.num_layers,
                                            transfer_fct=self.transfer_fct,
                                            act_out=None,
                                            reuse=self.reuse,
                                            kinit=self.kinit,
                                            bias_init=self.bias_init,
                                            drop_rate=self.dropout,
                                            prefix='post_en_',
                                            isConv=self.isConv)

            self.post_mean = Qlatent_x.output
            self.post_var = tf.nn.sigmoid(self.post_mean)

            self.latent = self.post_mean
            self.latent_batch = self.latent

        print('\n[*] Defining posterior decoder...')
        with tf.variable_scope('post_decoder', reuse=self.reuse):
            Px_latent = self.create_decoder(input_=self.latent_batch,
                                            hidden_dim=self.hidden_dim,
                                            output_dim=self.x_flat_dim,
                                            num_layers=self.num_layers,
                                            transfer_fct=self.transfer_fct,
                                            act_out=tf.nn.sigmoid,
                                            reuse=self.reuse,
                                            kinit=self.kinit,
                                            bias_init=self.bias_init,
                                            drop_rate=self.dropout,
                                            prefix='post_de_',
                                            isConv=self.isConv)

            self.x_recons_flat = Px_latent.output
        self.x_recons = tf.reshape(self.x_recons_flat, [-1, self.width, self.height, self.num_channels])

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

        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope('ae_loss', reuse=self.reuse):
            self.ae_loss = tf.add(tf.reduce_mean(self.reconstruction), self.l2*self.L2_loss, name='ae_loss')

        with tf.variable_scope('bayae_loss', reuse=self.reuse):
            bay_kl = losses.get_QP_kl(self.post_mean, self.post_var, self.prior_mean, self.prior_var)

            self.bayae_loss = tf.add(tf.cast(self.num_batches, 'float32')*self.ae_loss, bay_kl, name='bayae_loss')

        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.bayae_loss, global_step=self.global_step_tensor)

        self.losses = ['bayAE', 'AE', 'reconstruction', 'L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.bayae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, aeloss, recons, L2_loss
    
    def evaluate_epoch(self, session, x):
        tensors = [self.bayae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, aeloss, recons, L2_loss
