from enum import Enum
class Models:
    AE= 'AE'
    VAE= 'VAE'
    BVAE = 'BVAE'
    AnnealedVAE = 'AnnealedVAE'
    DIPIVAE = 'DIPIVAE'
    DIPIIVAE = 'DIPIIVAE'
    BTCVAE = 'BTCVAE'
    BayAE = 'BayAE'

# Stopping tolerance
tol = 1e-8
min_lr = 1e-8
epsilon = 1e-8
SAVE_EPOCH=20
COLAB_SAVE=50

import tensorflow as tf
config = dict()
config['model_name']= Models.AE
config['model_type']= Models.AE
config['dataset_name']= ''
config['latent_dim']= 15
config['num_layers']= 3
config['hidden_dim']= 100
config['l2']= 1e-6
config['batch_size']= 64
config['batch_norm']= True
config['learning_rate']= 1e-3
config['dropout']= 0.25
config['isConv']= False
config['epochs']= int(2e5)
config['restore']= False
config['plot']= False
config['colab']= False
config['colabpath']= ''
config['early_stopping']= True
config['log_dir']= 'log_dir'
config['checkpoint_dir']='checkpoint_dir'
config['summary_dir']='summary_dir'
config['act_out']= tf.nn.softplus
config['transfer_fct']=tf.nn.relu
config['kinit']=tf.contrib.layers.xavier_initializer()
config['bias_init']=tf.constant_initializer(0.0)
config['reuse']=False

config['latent_max']=1
config['latent_min']=1
config['latent_std']=1

####### Beta VAE
config['beta']=10.0

####### AnnealedVAE
config['ann_gamma']=100
config['c_max']=25
config['itr_thd']=1000

####### DIPIVAE
config['lambda_d']=10
config['d_factor']=10

####### BayesianVAE
#Monte Carlo samples
config['num_batches']=1000
config['MC_samples']=10



def get_model_name_AE(model, config):
    conv = 'Conv'  if config.isConv else ''
    model_name = model + '_' \
                 + conv + '_' \
                 +  config.dataset_name+ '_' \
                 + 'lat' + str(config.latent_dim) + '_' \
                 + 'h' + str(config.hidden_dim)  + '_' \
                 + 'lay' + str(config.num_layers)
    return model_name


def get_model_name(model, config):
    if model in [Models.AE, Models.VAE]:
        return get_model_name_AE(model, config)

    elif model in [Models.BVAE, Models.BTCVAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'b' + str(config.beta).replace('.','')

    elif model in [Models.AnnealedVAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'g' + str(config.ann_gamma)     + '_' \
                 + 'cmax' + str(config.c_max)  + '_' \
                 + 'ithd' + str(config.itr_thd)

    elif model in [Models.DIPIVAE, Models.DIPIIVAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'lmd' + str(config.lambda_d) + '_' \
                 + 'fact' + str(config.d_factor)

    elif model in [Models.BayAE]:
        return get_model_name_AE(model, config) + '_' \
                 + 'mc' + str(config.MC_samples)