
__author__      = "Khalid M. Kahloot"
__copyright__   = "Copyright 2019, Only for professionals"

""" 
------------------------------------------------------------------------------
AE.pytoencoder argument process
------------------------------------------------------------------------------
"""

import os
import sys

sys.path.append('..')
import copy
import numpy as np
import tensorflow as tf

from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
import utils.utils as utils
from utils.logger import Logger
from utils.early_stopping import EarlyStopping

import utils.constants as const
from graphs.AE_Factory import Factory
from base.base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler

class AE(BaseModel):
    '''
    ------------------------------------------------------------------------------
                                         SET ARGUMENTS
    -------------------------------------------------------------------------------
    '''
    def __init__(self, **kwrds):
        self.config = utils.Config(copy.deepcopy(const.config))
        for key in kwrds.keys():
            assert key in self.config.keys(), '{} is not a keyword, \n acceptable keywords: {}'.\
                format(key, self.config.keys())
            self.config[key] = kwrds[key]

        self.experiments_root_dir = 'experiments'
        utils.create_dirs([self.experiments_root_dir])
        self.config.model_name = const.get_model_name(self.config.model_type, self.config)
        self.config.checkpoint_dir = os.path.join(self.experiments_root_dir + "/" + self.config.checkpoint_dir + "/",
                                                  self.config.model_name)
        self.config.summary_dir = os.path.join(self.experiments_root_dir+"/"+self.config.summary_dir+"/", self.config.model_name)
        self.config.log_dir = os.path.join(self.experiments_root_dir + "/" + self.config.log_dir + "/",
                                               self.config.model_name)

        utils.create_dirs([self.config.checkpoint_dir, self.config.summary_dir, self.config.log_dir])
        load_config = {}
        try:
            load_config = utils.load_args(self.config.model_name, self.config.summary_dir)
            self.config.update(load_config)
            self.config.update({key: const.config[key] for key in ['kinit', 'bias_init', 'act_out', 'transfer_fct']})
            print('Loading previous configuration ...')
        except:
            print('Unable to load previous configuration ...')

        utils.save_args(self.config.dict(), self.config.model_name, self.config.summary_dir)

        if self.config.plot:
            self.latent_space_files = list()
            self.latent_space3d_files = list()
            self.recons_files = list()

        if hasattr(self.config, 'height'):
            try:
                self.config.restore = True
                self.build_model(self.config.height, self.config.width, self.config.num_channels)
            except:
                self.isBuild = False
        else:
            self.isBuild = False

    '''
    ------------------------------------------------------------------------------
                                         EPOCH FUNCTIONS
    -------------------------------------------------------------------------------
    '''
    def _train(self, data_train, session, logger):
        losses = list()
        for _ in tqdm(range(data_train.num_batches(self.config.batch_size))):
            batch_x = next(data_train.next_batch(self.config.batch_size))
            loss_curr = self.model_graph.train_epoch(session, batch_x)

            losses.append(loss_curr)

        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it,  summarizer='train', summaries_dict=summaries_dict)
        return losses

    def _evaluate(self, data_eval, session, logger):
        losses = list()
        for _ in tqdm(range(data_eval.num_batches(self.config.batch_size))):
            batch_x = next(data_eval.next_batch(self.config.batch_size))
            loss_curr = self.model_graph.evaluate_epoch(session, batch_x)

            losses.append(loss_curr)

        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it,  summarizer='evaluate', summaries_dict=summaries_dict)
        return losses

    def _evaluate_metric(self, data_eval, session, logger):
        losses = list()
        for _ in tqdm(range(data_eval.num_batches(self.config.batch_size))):
            batch_x = next(data_eval.next_batch(self.config.batch_size))
            loss_curr = self.model_graph.evaluate_epoch_metric(session, batch_x)

            losses.append(loss_curr)

        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it,  summarizer='evaluate', summaries_dict=summaries_dict)
        return losses

    def _evaluate_metric_supervised(self, data_eval, session, logger):
        losses = list()
        for _ in tqdm(range(data_eval.num_batches(self.config.batch_size))):
            batch_x, batch_y = next(data_eval.next_batch(self.config.batch_size, with_labels = True))
            loss_curr = self.model_graph.evaluate_epoch_metric_supervised(session, batch_x, batch_y)

            losses.append(loss_curr)

        losses = np.mean(np.vstack(losses), axis=0)

        cur_it = self.model_graph.global_step_tensor.eval(session)
        summaries_dict = dict(zip(self.model_graph.losses, losses))

        logger.summarize(cur_it,  summarizer='evaluate', summaries_dict=summaries_dict)
        return losses

    '''
    ------------------------------------------------------------------------------
                                         EPOCH FUNCTIONS
    -------------------------------------------------------------------------------
    '''

    def fit(self, X, y=None):
        print('\nProcessing data...')
        self.data_train, self.data_eval = utils.process_data(X, y)
        self.config['num_batches'] = self.data_train.num_batches(self.config.batch_size)

        if not self.isBuild:
            self.config.restore=True
            self.build_model(self.data_train.height, self.data_train.width, self.data_train.num_channels)
        else:
            assert (self.config.height == self.data_train.height) and (self.config.width == self.data_train.width) and \
                   (self.config.num_channels == self.data_train.num_channels), \
                    'Wrong dimension of data. Expected shape {}, and got {}'.format((self.config.height,self.config.width, \
                                                                                     self.config.num_channels), \
                                                                                    (self.data_train.height,
                                                                                     self.data_train.width, \
                                                                                     self.data_train.num_channels) \
                                                                                    )

        '''  -------------------------------------------------------------------------------
                                        TRAIN THE MODEL
        ------------------------------------------------------------------------------------- '''
        print('\nTraining a model...')

        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(222222)
            self.session = session
            logger = Logger(self.session, self.config.summary_dir)
            saver = tf.train.Saver()

            early_stopper = EarlyStopping(name='total loss', decay_fn=self.decay_fn)

            if(self.config.restore and self.load(self.session, saver) ):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()


            for cur_epoch in range(self.model_graph.cur_epoch_tensor.eval(self.session), self.config.epochs+1, 1):

                print('EPOCH: ', cur_epoch)
                self.current_epoch = cur_epoch

                losses_tr = self._train(self.data_train, self.session, logger)

                if np.isnan(losses_tr[0]):
                    print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                    for lname, lval in zip(self.model_graph.losses, losses_tr):
                        print(lname, lval)
                    sys.exit()

                losses_eval = self._evaluate(self.data_eval, self.session, logger)

                train_msg = 'TRAIN: \n'
                for lname, lval in zip(self.model_graph.losses, losses_tr):
                    train_msg += str(lname) + ': ' + str(lval) + ' | '

                eval_msg = 'EVALUATE: \n'
                for lname, lval in zip(self.model_graph.losses, losses_eval):
                    eval_msg += str(lname) + ': ' + str(lval) + ' | '

                print(train_msg)
                print(eval_msg)
                print()

                if (cur_epoch == 1) or ((cur_epoch % const.SAVE_EPOCH == 0) and (cur_epoch != 0)):
                    self.save(self.session, saver, self.model_graph.global_step_tensor.eval(self.session))
                    if self.config.plot:
                        self.reconst_samples_from_data(self.data_train, self.session, cur_epoch)

                self.session.run(self.model_graph.increment_cur_epoch_tensor)

                # Early stopping
                if (self.config.early_stopping and early_stopper.stop(losses_eval[0])):
                    print('Early Stopping!')
                    break

                if cur_epoch % const.COLAB_SAVE == 0:
                    if self.config.colab:
                        self.push_colab()

            self.save(self.session, saver, self.model_graph.global_step_tensor.eval(self.session))
            if self.config.plot:
                self.reconst_samples_from_data(self.data_train, self.session, cur_epoch)

            if self.config.colab:
                self.push_colab()
            z = self.encode(self.data_train.x)
            self.config.latent_max = z.max().compute()
            self.config.latent_min = z.min().compute()
            self.config.latent_std = z.std().compute()
            del z
        return


    '''  ------------------------------------------------------------------------------
                                     SET NETWORK PARAMS
     ------------------------------------------------------------------------------ '''
    def build_model(self, height, width, num_channels):
        self.config['height'] = height
        self.config['width'] = width
        self.config['num_channels'] = num_channels

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model_graph = Factory(self.config.__dict__)
            print(self.model_graph)

            self.trainable_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print('\nNumber of trainable paramters', self.trainable_count)
            self.test_graph()

        '''  -------------------------------------------------------------------------------
                        GOOGLE COLAB 
        ------------------------------------------------------------------------------------- '''
        if self.config.colab:
            self.push_colab()
            self.config.push_colab = self.push_colab

        self.isBuild=True
        utils.save_args(self.config.dict(), self.config.model_name, self.config.summary_dir)


    def test_graph(self):
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(222222)
            self.session = session
            logger = Logger(self.session, self.config.summary_dir)
            saver = tf.train.Saver()

            if (self.config.restore and self.load(self.session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            print('random sample batch ...')
            samples = self.model_graph.sample(session)
            print('random sample shape {}'.format(samples.shape))

    def run_metrics(self):
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(222222)
            self.session = session
            logger = Logger(self.session, self.config.summary_dir)
            saver = tf.train.Saver()

            if (self.config.restore and self.load(self.session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            print('random sample batch ...')
            metric  = self._evaluate_metric(self.data_eval, self.session, logger)
            print('Unsupervised metric{}'.format(metric))
            metric2 = self._evaluate_metric_supervised(self.data_eval, self.session, logger)
            print('Supervised metric  {}'.format(metric2))

    def reconst_samples_out_data(self):
        with tf.Session(graph=self.graph) as session:
            tf.set_random_seed(222222)
            self.session = session
            logger = Logger(self.session, self.config.summary_dir)
            saver = tf.train.Saver()

            if (self.config.restore and self.load(self.session, saver)):
                num_epochs_trained = self.model_graph.cur_epoch_tensor.eval(self.session)
                print('EPOCHS trained: ', num_epochs_trained)
            else:
                print('Initializing Variables ...')
                tf.global_variables_initializer().run()

            print('random sample batch ...')
            samples = self.model_graph.sample(session)
        scaler = MinMaxScaler()
        return scaler.fit_transform(samples.flatten().reshape(-1, 1).astype(np.float32)).reshape(samples.shape)


