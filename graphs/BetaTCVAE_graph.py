"""
BetaTCVAE_graph.py:
Tensorflow Graph for the Beta-TC-Variational Autoencoder
"""
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"
__paper__   = "https://arxiv.org/pdf/1802.04942"

import math
import tensorflow as tf
import numpy as np
import utils.constants as const
from base.base_graph import BaseGraph
import base.losses as losses

'''
This is the Main BTCVAEGraph.
'''
class BTCVAEGraph(BaseGraph):
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

        with tf.variable_scope('bvae_loss', reuse=self.reuse):
            regularizer = tf.multiply(self.beta , self.kl_loss_m)
            self.bvae_loss = tf.add(self.ae_loss, regularizer)

        with tf.variable_scope('btcvae_loss', reuse=self.reuse):
            """
            Based on Equation 4 with alpha = gamma = 1 of "Isolating Sources of Disentanglement in Variational
            Autoencoders"
            (https: // arxiv.org / pdf / 1802.04942).
            If alpha = gamma = 1, Eq 4 can be
            written as ELBO + (1 - beta) * TC.
            """
            tc =  tf.multiply(self.beta-1, self.total_correlation(self.latent_batch, self.encoder_mean, \
                                                                  self.encoder_logvar))
            regularizer =  tf.add(self.kl_loss_m, tc)
            self.btcvae_loss = tf.add(self.ae_loss, regularizer)

        with tf.variable_scope("optimizer" ,reuse=self.reuse):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_step = self.optimizer.minimize(self.btcvae_loss, global_step=self.global_step_tensor)

        self.losses = ['Beta-TCVAE', 'Beta-VAE', 'VAE', 'AE', 'reconstruction', 'L2']

    '''  
    ------------------------------------------------------------------------------
                                     FIT & EVALUATE TENSORS
    ------------------------------------------------------------------------------ 
    '''
    def train_epoch(self, session, x):
        tensors = [self.train_step, self.btcvae_loss, self.bvae_loss, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        _, loss, bvaeloss, vaeloss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, bvaeloss, vaeloss, aeloss, recons, L2_loss
    
    def evaluate_epoch(self, session, x):
        tensors = [self.btcvae_loss, self.bvae_loss, self.vae_loss, self.ae_loss, self.loss_reconstruction_m, self.L2_loss]
        feed_dict = {self.x_batch: x}
        loss, bvaeloss, vaeloss, aeloss, recons, L2_loss  = session.run(tensors, feed_dict=feed_dict)
        return loss, bvaeloss, vaeloss, aeloss, recons, L2_loss


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
                                          GRAPH OPERATIONS
     ------------------------------------------------------------------------------ 
     '''

    def gaussian_log_density(self, samples, mean, log_var):
        pi = tf.constant(math.pi)
        normalization = tf.log(2. * pi)
        inv_sigma = tf.exp(-log_var)
        tmp = (samples - mean)
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def total_correlation(self, latent, latent_mean, latent_logvar):
        """Estimate of total correlation on a batch.
        We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
        log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
        for the minimization. The constant should be equal to (num_latents - 1) *
        log(batch_size * dataset_size)
        Args:
          z: [batch_size, num_latents]-tensor with sampled representation.
          z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
          z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
        Returns:
          Total correlation estimated on a batch.
        """
        # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
        # tensor of size [batch_size, batch_size, num_latents]. In the following
        # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
        log_qlatent_prob = self.gaussian_log_density(
            tf.expand_dims(latent, 1), tf.expand_dims(latent_mean, 0),
            tf.expand_dims(latent_logvar, 0))
        # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
        # + constant) for each sample in the batch, which is a vector of size
        # [batch_size,].
        log_qlatent_product = tf.reduce_sum(tf.reduce_logsumexp(log_qlatent_prob, axis=1, keepdims=False),
                                                                             axis=1,
                                                                             keepdims=False)
        # Compute log(q(latent(x_j))) as log(sum_i(q(latent(x_j)|x_i))) + constant =
        # log(sum_i(prod_l q(latent(x_j)_l|x_i))) + constant.
        log_qlatent = tf.reduce_logsumexp(tf.reduce_sum(log_qlatent_prob, axis=2, keepdims=False),
                                                                     axis=1,
                                                                     keepdims=False)
        return tf.reduce_mean(log_qlatent - log_qlatent_product)