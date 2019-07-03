"""
AE_graph.py:
Tensorflow Graph for the Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://arxiv.org/pdf/1404.7828.pdf, http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf"


import tensorflow as tf
import numpy as np
import utils.constants as const
from base.base_graph import BaseGraph
import base.losses as losses

'''
This is the Main AEGraph.
'''
class AEGraph(BaseGraph):
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
        print('\n[*] Defining encoder...')
        with tf.variable_scope('encoder', reuse=self.reuse):
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
                            prefix='en_',
                            isConv=self.isConv)
        
            self.latent = Qlatent_x.output
            self.latent_batch = self.latent
            
        print('\n[*] Defining decoder...')
        with tf.variable_scope('decoder', reuse=self.reuse):
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
                                            prefix='de_',
                                            isConv=self.isConv)
        
            self.x_recons_flat = Px_latent.output
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

        with tf.variable_scope("L2_loss", reuse=self.reuse):
            tv = tf.trainable_variables()
            self.L2_loss = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        
        with tf.variable_scope('ae_loss', reuse=self.reuse):
            self.ae_loss = tf.add(tf.reduce_mean(self.reconstruction), self.l2*self.L2_loss, name='ae_loss')

        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.ae_loss, global_step=self.global_step_tensor)

        self.losses = ['AE', 'reconstruction', 'L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, L2_loss
    
    def evaluate_epoch(self, session, x):
        tensors = [self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, recons, L2_loss


    '''  
    ------------------------------------------------------------------------------
                                     GENERATE LATENT and RECONSTRUCT
    ------------------------------------------------------------------------------ 
    '''
    def reconst_loss(self, session, x):
        tensors= [self.reconstruction] 
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
        random_latent = session.run(tf.random_normal((self.batch_size, self.latent_dim), -1, 1, dtype=tf.float32))
        tensors = [self.x_recons]
        feed_dict = {self.latent_batch: random_latent}
        return session.run(tensors, feed_dict=feed_dict)[0]

