"""
DIPIIVAE_graph.py:
Tensorflow Graph for the DIPII Variational Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://openreview.net/pdf?id=H1kG7GZAW"

import tensorflow as tf
import numpy as np
import utils.constants as const
from base.base_graph import BaseGraph
import base.losses as losses

'''
This is the Main DIPIVAEGraph.
'''
class DIPIIVAEGraph(BaseGraph):
    def __init__(self, configuration):
        BaseGraph.__init__(self)
        self.__dict__.update(configuration)
        self.x_flat_dim = self.width * self.height * self.num_channels
        self.diag = self.d_factor * self.lambda_d
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

        with tf.variable_scope('dipvae_loss', reuse=self.reuse):
            regularize = tf.add(self.regularizer(self.encoder_mean, self.encoder_logvar), self.kl_loss_m)
            self.dipvae_loss = tf.add(self.ae_loss, regularize)

        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.dipvae_loss, global_step=self.global_step_tensor)

        self.losses = ['DIPVAE', 'VAE', 'AE', 'reconstruction', 'L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.dipvae_loss, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, vaeloss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, vaeloss, aeloss, recons, L2_loss
    
    def evaluate_epoch(self, session, x):
        tensors = [self.dipvae_loss, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, vaeloss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, vaeloss, aeloss, recons, L2_loss


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

    '''  
    ------------------------------------------------------------------------------
                                         DIP OPERATIONS
    ------------------------------------------------------------------------------ 
    '''

    def compute_covariance_latent_mean(self, latent_mean):
        """
        :param latent_mean:
        :return:
        Computes the covariance of latent_mean.
        Uses cov(latent_mean) = E[latent_mean*latent_mean^T] - E[latent_mean]E[latent_mean]^T.
        Args:
          latent_mean: Encoder mean, tensor of size [batch_size, num_latent].
        Returns:
          cov_latent_mean: Covariance of encoder mean, tensor of size [latent_dim, latent_dim].
        """
        exp_latent_mean_latent_mean_t = tf.reduce_mean(
            tf.expand_dims(latent_mean, 2) * tf.expand_dims(latent_mean, 1), axis=0)
        expectation_latent_mean = tf.reduce_mean(latent_mean, axis=0)

        cov_latent_mean = tf.subtract(exp_latent_mean_latent_mean_t,
          tf.expand_dims(expectation_latent_mean, 1) * tf.expand_dims(expectation_latent_mean, 0))
        return cov_latent_mean

    def regularize_diag_off_diag_dip(self, covariance_matrix, lambda_d, diag):
        """
        Compute on and off diagonal regularizers for DIP-VAE models.
        Penalize deviations of covariance_matrix from the identity matrix. Uses
        different weights for the deviations of the diagonal and off diagonal entries.
        Args:
            covariance_matrix: Tensor of size [num_latent, num_latent] to regularize.
            lambda_d: Weight of penalty for off diagonal elements.
            diag: Weight of penalty for diagonal elements.
        Returns:
            dip_regularizer: Regularized deviation from diagonal of covariance_matrix.
        """
        covariance_matrix_diagonal = tf.diag_part(covariance_matrix)
        covariance_matrix_off_diagonal = covariance_matrix - tf.diag(covariance_matrix_diagonal)
        dip_regularizer = tf.add(
            lambda_d * tf.reduce_sum(covariance_matrix_off_diagonal ** 2),
            diag * tf.reduce_sum((covariance_matrix_diagonal - 1) ** 2))

        return dip_regularizer

    def regularizer(self, latent_mean, latent_logvar):
        cov_latent_mean = self.compute_covariance_latent_mean(latent_mean)
        cov_enc = tf.matrix_diag(tf.exp(latent_logvar))
        expectation_cov_enc = tf.reduce_mean(cov_enc, axis=0)
        cov_latent = expectation_cov_enc + cov_latent_mean
        cov_dip_regularizer = self.regularize_diag_off_diag_dip(cov_latent, self.lambda_d, self.diag)

        return cov_dip_regularizer