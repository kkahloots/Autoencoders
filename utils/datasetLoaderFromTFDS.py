#Requirements needs to be updated with the new addition of "tensorflow_datasets"
#!pip install tensorflow_datasets
import tensorflow_datasets as tfds
import numpy as np
import os
from pathlib import Path

def giveDatasetInfo(  datasetName: str):
    dataset, info = tfds.load(datasetName, with_info=True)
    print(info)

def loadDataset(  datasetName: str, numberOfExamples : int):
    print("Started loading into Numpy")
    if datasetName == 'smallnorb':
       X, y =  loadSmallNorb(datasetName, numberOfExamples)
    elif datasetName == "shapes3d":
       X, y = loadShapes3D(datasetName, numberOfExamples)
    print("Finished loading into Numpy")
    return X, y

def loadSmallNorb(  datasetName: str, numberOfExamples : int):
    dataset = tfds.load(datasetName, with_info=False)
    train_ds = dataset['train'].take(numberOfExamples)
    y = None
    X = None
    for example in tfds.as_numpy( train_ds):
        if (y is None):
            y = np.array([[example["instance"], example["label_azimuth"], example["label_category"],
                                 example['label_elevation'], example['label_lighting']]])
            y = np.append(y, [
                [example["instance"], example["label_azimuth"], example["label_category"], example['label_elevation'],
                 example['label_lighting']]], 0)
        else:
            y = np.append(y, [
                [example["instance"], example["label_azimuth"], example["label_category"], example['label_elevation'],
                 example['label_lighting']]], 0)
            y = np.append(y, [
                [example["instance"], example["label_azimuth"], example["label_category"], example['label_elevation'],
                 example['label_lighting']]], 0)
        if (X is None):
            X = np.array([example["image"], example["image2"]])
        else:
            X = np.append(X, [example["image"], example["image2"]], 0)
    return X, y

def loadShapes3D(  datasetName: str, numberOfExamples : int):
    dataset = tfds.load(datasetName, with_info=False)
    train_ds = dataset['train'].take(numberOfExamples)
    y = None
    X = None
    for example in tfds.as_numpy(train_ds):
        if (y is None):
            y = np.array([[example["label_floor_hue"], example["label_object_hue"], example["label_orientation"],
                           example['label_scale'], example['label_shape'], example["label_wall_hue"],
                           example["value_floor_hue"], example["value_object_hue"], example["value_orientation"],
                           example["value_scale"], example["value_shape"], example["value_wall_hue"]]])

        else:
            y = np.append(y, [[example["label_floor_hue"], example["label_object_hue"], example["label_orientation"],
                           example['label_scale'], example['label_shape'], example["label_wall_hue"],
                           example["value_floor_hue"], example["value_object_hue"], example["value_orientation"],
                           example["value_scale"], example["value_shape"], example["value_wall_hue"]]], 0)
        if (X is None):
            X = np.array([example["image"]])
        else:
            X = np.append(X, [example["image"]], 0)
    return X, y


def save_to_npz(datasetName: str, X, Y):
    print('Started Saving arrays to npz')
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    file_name = "{0}.npz".format(datasetName)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.savez(os.path.join(BASE_PATH, file_name), X=X, Y=Y)
    print('Finished Saving arrays to npz')


def load_from_npz(datasetName: str):
    print('Started loading arrays from npz')
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    file_name = "{0}.npz".format(datasetName)
    data = np.load(os.path.join(BASE_PATH, file_name))
    print('Finished loading arrays from npz')
    return data['X'], data['Y']

def download_npz(datasetName: str):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    g_login = GoogleAuth()

    #g_login.Authorize()
    g_login.LocalWebserverAuth()
    drive = GoogleDrive(g_login)
    return


