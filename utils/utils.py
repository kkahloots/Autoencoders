import sys
import os
sys.path.append('..')

import random
import json

import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from skimage.color import grey2rgb

from utils.dataset import Dataset


'''  ------------------------------------------------------------------------------
                                    DATA METHODS
 ------------------------------------------------------------------------------ '''
scalar = None

def prepare_dataset(X):
    len_ = X.shape[0]
    shape_ = X.shape

    d = int(np.sqrt(X.flatten().reshape(X.shape[0], -1).shape[1]))
    print(shape_)
    if len(shape_) == 4 and shape_[3] == 3:
        d = int(np.sqrt(X.flatten().reshape(X.shape[0], -1).shape[1] / 3))
        X = np.reshape(X, [-1, d, d, 3])
    elif len(shape_) == 4 and shape_[3] == 1:
        d = int(np.sqrt(X.flatten().reshape(X.shape[0], -1).shape[1] ))
        X = np.reshape(X, [-1, d, d, 1])
    elif d == shape_[1] and len(shape_) == 3:
        X = np.array(list(map(lambda x: grey2rgb(x), X)), dtype=np.float32)
        X = np.reshape(X, [-1, d, d, 3])

    else:
        r = d**2 - X.shape[1]
        train_padding = np.zeros((shape_[0], r))
        X = np.vstack([X, train_padding])

        X = np.reshape(X, [-1, d, d])
        X = np.array(list(map(lambda x: grey2rgb(x), X)), dtype=np.float32)

    print('Scaling dataset ... ')
    if scalar is not None:
        X = scaler.transform(X.flatten().reshape(-1, 1).astype(np.float32)).reshape(X.shape)
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X.flatten().reshape(-1, 1).astype(np.float32)).reshape(X.shape)
    print('Creating dask array ... ')
    return da.from_array(X, chunks=100)#da.array(X)#

def process_data(X, y=None, test_size=0.20, dummies=False):
    if y is None:
        km = dask_ml.cluster.KMeans(n_clusters=10, init_max_iter=100)
        km.fit(X.flatten().reshape(-1, 1))
        y = km.labels_
    y_uniqs = np.unique(y)

    len_ = X.shape[0]
    X = prepare_dataset(X)

    if dummies:
        y = dd.get_dummies(y)

    shape_ = list(X.shape[1:])

    samples = list()
    samples_labels = list()
    print('Preparing samples ...')
    for _ in range(2):
        for y_uniq in y_uniqs:
            sample = list()
            label = list()
            for xa, ya in zip(chunks(X, 100),chunks(y, 100)):
                try:
                    print("hello")
                    sample.append([xa[ya == y_uniq][random.randint(0, len(xa[ya == y_uniq]) - 1)]])
                    label.append(y_uniq)
                    if len(sample) >= 10:
                        break
                except Exception as e:
                    print(e)
                    pass
            samples += sample
            samples_labels += label
    samples = da.vstack(samples)
    samples_labels = da.vstack(samples_labels)

    X_train, X_test, y_train, y_test = train_test_split(X.flatten().reshape(len_, -1), y, test_size=test_size,
                                                        random_state=4891)

    X_train = X_train.reshape([X_train.shape[0]] + shape_)
    X_test = X_test.reshape([X_test.shape[0]] + shape_)

    print('Training dataset shape: ', X_train.shape)
    print('Validation dataset shape: ', X_test.shape)

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    train_dataset.samples = samples
    train_dataset.samples_labels = samples_labels

    print('Sample dataset shape: ', train_dataset.samples.shape)
    return train_dataset, test_dataset


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_args(args, exp_name, summary_dir):
    print('Saving Model Arguments ...')
    my_file = summary_dir + '/' + exp_name + '.json'
    with open(my_file, 'w') as fp:
        json.dump(args, fp)

def load_args(exp_name, summary_dir):
    my_file = summary_dir + '/' + exp_name + '.json'
    with open(my_file, 'r') as fp:
        args = json.load(fp)
    return args


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        sys.exit(-1)

from types import FunctionType
class Config:
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def update(self, newvals):
        self.__dict__.update(newvals)

    def keys(self):
        keys = list()
        for key, item in self.__dict__.items():
            if type(item) != FunctionType:
                keys.append(key)
        return keys

    def dict(self):
        keys = list()
        items = list()
        for key, item in self.__dict__.items():
            if key not in ['kinit', 'bias_init', 'act_out', 'transfer_fct']:
                if type(item) != FunctionType:
                    keys.append(key)
                    items.append(item)
        return dict(zip(keys, items))
