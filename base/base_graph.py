

import tensorflow as tf
from networks.dense_net import DenseNet
from networks.conv_net import ConvNet3
from networks.deconv_net import DeconvNet3

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



