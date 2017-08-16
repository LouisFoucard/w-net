from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.image as mpimg
import os
import numpy as np
from tqdm import tqdm


def get_data_generators(train_folder, val_folder, img_rows=128, img_cols=224, batch_size=16, shuffle=True):

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator1 = train_datagen.flow_from_directory(
        train_folder,
        target_size=(img_rows, 2 * img_cols),
        batch_size=batch_size,
        seed=10,
        shuffle=shuffle,
        classes=None,
        class_mode=None)

    train_generator2 = train_datagen.flow_from_directory(
        train_folder,
        target_size=(img_rows, 2 * img_cols),
        batch_size=batch_size,
        seed=10,
        shuffle=shuffle,
        classes=None,
        class_mode=None)

    validation_generator1 = val_datagen.flow_from_directory(
        val_folder,
        target_size=(img_rows, 2 * img_cols),
        batch_size=batch_size,
        seed=10,
        shuffle=shuffle,
        classes=None,
        class_mode=None)

    validation_generator2 = val_datagen.flow_from_directory(
        val_folder,
        target_size=(img_rows, 2 * img_cols),
        batch_size=batch_size,
        seed=10,
        shuffle=shuffle,
        classes=None,
        class_mode=None)

    def train_generator_func():
        while True:
            X = train_generator1.next()
            Y1 = train_generator2.next()
            yield X, [Y1, np.zeros(shape=(Y1.shape[0], img_rows - 4, img_cols - 4)),
                      np.zeros(shape=(Y1.shape[0], img_rows - 4, img_cols - 4))]

    def val_generator_func():
        while True:
            X = validation_generator1.next()
            Y1 = validation_generator2.next()
            yield X, [Y1, np.zeros(shape=(Y1.shape[0], img_rows - 4, img_cols - 4)),
                      np.zeros(shape=(Y1.shape[0], img_rows - 4, img_cols - 4))]

    train_generator = train_generator_func()
    val_generator = val_generator_func()

    training_samples = len(train_generator1.filenames)
    val_samples = len(validation_generator1.filenames)

    return train_generator, val_generator, training_samples, val_samples


def clean_directory(path):
    for img_file in tqdm(glob(path)):
        try:
            mpimg.imread(img_file)
        except Exception as e:
            print('removing ' + os.path.join(path, img_file))
            os.remove(os.path.join(path, img_file))


