#Requirements needs to be updated with the new addition of "tensorflow_datasets"
#!pip install tensorflow_datasets
import tensorflow_datasets as tfds
import numpy as np
import os
import cv2

def giveDatasetInfo(  datasetName: str):
    dataset, info = tfds.load(datasetName, with_info=True)
    print(info)

def loadDataset(  datasetName: str, numberOfExamples : int):
    print("Started loading into Numpy")
    if datasetName == 'smallnorb':
       X, y = loadSmallNorb(datasetName, numberOfExamples)
    elif datasetName == "shapes3d":
       X, y = loadShapes3D(datasetName, numberOfExamples)
    elif datasetName == "celeb_a":
       print("Started loading celeb_a")
       X, y, L = loadCelebA(datasetName, numberOfExamples)
       return X, y, L
    elif datasetName == 'dsprites':
       X, y = loadDStripes(datasetName, numberOfExamples)
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


def loadCelebA(  datasetName: str, numberOfExamples : int):
    dataset = tfds.load(datasetName, with_info=False)
    train_ds = dataset['train'].take(numberOfExamples)
    test_ds = None
    if numberOfExamples > 162770:
        test_ds = dataset['test'].take(numberOfExamples - 162770) # 162770 train split size
    X, y, L = celeb_a_iterator(datasetName,train_ds)

    if test_ds != None:
        print('Started loading test split')
        X, y, L = celeb_a_iterator(datasetName, train_ds)
    return X, y, L

def celeb_a_iterator(datasetName, train_ds):
    y = None
    X = None
    L = None
    for index, iterator in enumerate(tfds.as_numpy(train_ds)):
        if (y is None):
            example = iterator['attributes']
            y = np.array([[
                example['5_o_Clock_Shadow'],
                example['Arched_Eyebrows'],
                example['Attractive'],
                example['Bags_Under_Eyes'],
                example['Bald'],
                example['Bangs'],
                example['Big_Lips'],
                example['Big_Nose'],
                example['Black_Hair'],
                example['Blond_Hair'],
                example['Blurry'],
                example['Brown_Hair'],
                example['Bushy_Eyebrows'],
                example['Chubby'],
                example['Double_Chin'],
                example['Eyeglasses'],
                example['Goatee'],
                example['Gray_Hair'],
                example['Heavy_Makeup'],
                example['High_Cheekbones'],
                example['Male'],
                example['Mouth_Slightly_Open'],
                example['Mustache'],
                example['Narrow_Eyes'],
                example['No_Beard'],
                example['Oval_Face'],
                example['Pale_Skin'],
                example['Pointy_Nose'],
                example['Receding_Hairline'],
                example['Rosy_Cheeks'],
                example['Sideburns'],
                example['Smiling'],
                example['Straight_Hair'],
                example['Wavy_Hair'],
                example['Wearing_Earrings'],
                example['Wearing_Hat'],
                example['Wearing_Lipstick'],
                example['Wearing_Necklace'],
                example['Wearing_Necktie'],
                example['Young']

            ]])
        else:
            example = iterator['attributes']
            y = np.append(y,
                          [[
                              example['5_o_Clock_Shadow'],
                              example['Arched_Eyebrows'],
                              example['Attractive'],
                              example['Bags_Under_Eyes'],
                              example['Bald'],
                              example['Bangs'],
                              example['Big_Lips'],
                              example['Big_Nose'],
                              example['Black_Hair'],
                              example['Blond_Hair'],
                              example['Blurry'],
                              example['Brown_Hair'],
                              example['Bushy_Eyebrows'],
                              example['Chubby'],
                              example['Double_Chin'],
                              example['Eyeglasses'],
                              example['Goatee'],
                              example['Gray_Hair'],
                              example['Heavy_Makeup'],
                              example['High_Cheekbones'],
                              example['Male'],
                              example['Mouth_Slightly_Open'],
                              example['Mustache'],
                              example['Narrow_Eyes'],
                              example['No_Beard'],
                              example['Oval_Face'],
                              example['Pale_Skin'],
                              example['Pointy_Nose'],
                              example['Receding_Hairline'],
                              example['Rosy_Cheeks'],
                              example['Sideburns'],
                              example['Smiling'],
                              example['Straight_Hair'],
                              example['Wavy_Hair'],
                              example['Wearing_Earrings'],
                              example['Wearing_Hat'],
                              example['Wearing_Lipstick'],
                              example['Wearing_Necklace'],
                              example['Wearing_Necktie'],
                              example['Young']

                          ]], 0)
        if (X is None):
            example = iterator
            X = np.array([example["image"]])
        else:
            example = iterator
            X = np.append(X, [example["image"]], 0)

        if (L is None):
            example = iterator['landmarks']
            L = np.array([[
                example['lefteye_x'],
                example['lefteye_y'],
                example['leftmouth_x'],
                example['leftmouth_y'],
                example['nose_x'],
                example['nose_y'],
                example['righteye_x'],
                example['righteye_y'],
                example['rightmouth_x'],
                example['rightmouth_y']
            ]])
        else:
            example = iterator['landmarks']
            L = np.append(L,
                          [[
                              example['lefteye_x'],
                              example['lefteye_y'],
                              example['leftmouth_x'],
                              example['leftmouth_y'],
                              example['nose_x'],
                              example['nose_y'],
                              example['righteye_x'],
                              example['righteye_y'],
                              example['rightmouth_x'],
                              example['rightmouth_y']

                          ]], 0)
        if index % 10000 == 0 and index != 0:
            print('Loaded {} 10000 pictures'.format(index / 10000))
            print('Resizing images to 64x64')
            X_res = None
            for img in X:
                img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                if (X_res is None):
                    X_res = np.array([img])
                else:
                    X_res = np.append(X_res, [img], 0)
                if X_res.shape[0] % 1000 == 0:
                    print('Finished resizing{} out of 10000'.format(X_res.shape[0]))
            save_to_npz(datasetName + '_{}_64x64'.format(index / 10000), X_res, y, L)
            y = None
            X = None
            L = None
    print('Loaded last {} pictures'.format(index + 1))
    print('Resizing images to 64x64')
    X_res = None
    for img in X:
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        if (X_res is None):
            X_res = np.array([img])
        else:
            X_res = np.append(X_res, [img], 0)
        if X_res.shape[0] % 1000 == 0:
            print('Finished resizing{} out of 10000'.format(X_res.shape[0]))
    save_to_npz(datasetName + '_{}_64x64'.format(index + 1), X_res, y, L)
    return X_res, y, L

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def loadDStripes(  datasetName: str, numberOfExamples : int):
    dataset = tfds.load(datasetName, with_info=False)
    train_ds = dataset['train'].take(numberOfExamples)
    y = None
    X = None
    for example in tfds.as_numpy(train_ds):
        if y is None:
            y = np.array([[
                example['label_orientation'],
                example['label_scale'],
                example['label_shape'],
                example['label_x_position'],
                example['label_y_position'],
                example['value_orientation'],
                example['value_scale'],
                example['value_shape'],
                example['value_x_position'],
                example['value_y_position']
            ]])
        else:
            y = np.append(y,
                          [[
                              example['label_orientation'],
                              example['label_scale'],
                              example['label_shape'],
                              example['label_x_position'],
                              example['label_y_position'],
                              example['value_orientation'],
                              example['value_scale'],
                              example['value_shape'],
                              example['value_x_position'],
                              example['value_y_position']
                          ]], 0)
        if (X is None):
            X = np.array([example["image"]])
        else:
            X = np.append(X, [example["image"]], 0)

    return X, y

def save_to_npz(datasetName: str, X, Y, L=None):
    print('Started Saving arrays to npz')
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    file_name = "{0}.npz".format(datasetName)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.savez(os.path.join(BASE_PATH, file_name), X=X, Y=Y, L=L)
    print('Finished Saving arrays to npz')


def load_from_npz(datasetName: str):
    print('Started loading arrays from npz')
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    file_name = "{0}.npz".format(datasetName)
    data = np.load(os.path.join(BASE_PATH, file_name))
    print('Finished loading arrays from npz')
    if 'celeb_a' in datasetName:
        return data['X'], data['Y'], data['L']
    return data['X'], data['Y']

def resize_pictures(filename: str):
    print('Started resizing file {} from npz'.format(filename))
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    file_name = "{0}.npz".format(filename)
    data = np.load(os.path.join(BASE_PATH, file_name))

    x_orig = data['X']
    X = None
    for img in x_orig:
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        if (X is None):
            X = np.array([img])
        else:
            X = np.append(X, [img], 0)
        if X.shape[0] % 1000 == 0:
            print(X.shape[0])
    print('Finished resizing file {} from npz'.format(filename))
    return X, data['Y'], data['L']

def resize_all_npz(datasetName: str):
    print('Started loading arrays from npz')
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    directory = os.listdir(BASE_PATH)
    for fname in directory:
        if os.path.isfile(BASE_PATH + os.sep + fname) and datasetName in fname:
            X, Y, L = resize_pictures(os.path.splitext(fname)[0])
            new_name = os.path.splitext(fname)[0] + '_64x64'
            print('Saving cropped npz {}'.format(new_name))
            save_to_npz(new_name, X, Y, L)

def combine_all_cropped(datasetName: str):
    print('Started loading arrays from npz')
    BASE_PATH = os.path.join(os.getcwd() + '\\npz_data\\')
    directory = os.listdir(BASE_PATH)
    X_comb = None
    Y_comb = None
    L_comb = None
    for fname in directory:
        if os.path.isfile(BASE_PATH + os.sep + fname) and datasetName in fname and '_64x64'in fname:
            print('Started combining npz {}'.format(fname))
            data = np.load(BASE_PATH + fname)
            X, Y, L = data['X'], data['Y'], data['L']
            if X_comb is None :
                X_comb, Y_comb, L_comb = X, Y, L
            else:
                X_comb = np.append(X_comb, X, 0)
                Y_comb = np.append(Y_comb, Y, 0)
                L_comb = np.append(L_comb, L, 0)
            print('Finished combining npz {}'.format(fname))
    new_name = BASE_PATH + '{}_combined-64x64'.format(datasetName)
    print('Saving Combined cropped npz {}'.format(new_name))
    save_to_npz(new_name, X_comb, Y_comb, L_comb)
    print('Finished saving Combined cropped npz {}'.format(new_name))