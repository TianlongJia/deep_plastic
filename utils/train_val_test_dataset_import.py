import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import Rescaling


def import_dataset_train_val(path_train, path_val, image_height, image_width, batch_size):

    # Creating a training dataset
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        path_train,
        labels='inferred',
        # subset='training',
        label_mode='int',
        color_mode='rgb',
        image_size=(image_height, image_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        # validation_split=val_split,
        interpolation='bilinear'
    )


    # Creating a validation dataset
    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        path_val,
        labels='inferred',
        # subset='validation',
        label_mode='int',
        color_mode='rgb',
        image_size=(image_height, image_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        # validation_split=val_split,
        interpolation='bilinear'
    )


    return ds_train, ds_val


def import_dataset_test(path_test, image_height, image_width, batch_size):
    ds_test = tf.keras.preprocessing.image_dataset_from_directory(
        path_test,
        labels='inferred',
        # subset='validation',
        label_mode='int',
        color_mode='rgb',
        image_size=(image_height, image_width),
        batch_size=batch_size,
        shuffle=True,
        seed=123,
        # validation_split=test_split,
        interpolation='bilinear'
    )

    
    return ds_test
