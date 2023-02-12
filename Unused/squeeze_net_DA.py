import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.imagenet_utils import obtain_input_shape
from keras import backend as K
import utils.train_val_test_dataset_import as tvt
import utils.class_imbalances as ci
import models.TL_models as TL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils.confusion_matrix as CM
from sklearn.metrics import confusion_matrix
import sklearn
import utils.plots as PLT

image_height=224
image_width=224
batch_size=16
nr_epochs=50
path_train=[r'C:\Users\Andre\Desktop\Data MSc thesis\Training data\train_27m_0deg_center_bank',
            r'C:\Users\Andre\Desktop\Data MSc thesis\Training data\Data Aug NEW\train_27m_0deg_center_bank_DA_HV',
            r'C:\Users\Andre\Desktop\Data MSc thesis\Training data\Data Aug NEW\train_27m_0deg_center_bank_DA_DARK',
            r'C:\Users\Andre\Desktop\Data MSc thesis\Training data\Data Aug NEW\train_27m_0deg_center_bank_DA_BR',
            r'C:\Users\Andre\Desktop\Data MSc thesis\Training data\Data Aug NEW\train_27m_0deg_center_bank_DA_NI']

path_val=r'C:\Users\Andre\Desktop\Data MSc thesis\Test data\test_27m_0deg_center_bank_VAL'
path_test=r'C:\Users\Andre\Desktop\Data MSc thesis\Test data\test_27m_0deg_center_bank_TEST'

ds_train, ds_val = tvt.import_dataset_train_val(path_train[0], path_val, image_height, image_width, batch_size)
ds_train_HV, ds_val_HV = tvt.import_dataset_train_val(path_train[1], path_val, image_height, image_width, batch_size)
ds_train_DARK, ds_val_DARK = tvt.import_dataset_train_val(path_train[2], path_val, image_height, image_width, batch_size)
ds_train_BR, ds_val_BR = tvt.import_dataset_train_val(path_train[3], path_val, image_height, image_width, batch_size)
ds_train_NI, ds_val_NI = tvt.import_dataset_train_val(path_train[4], path_val, image_height, image_width, batch_size)

ds_test = tvt.import_dataset_test(path_test, image_height, image_width, batch_size)

AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)

class_weights_train = ci.class_weights(path_train[0])

save_model_path_fromscratch_DA = [r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights_with_DA\dense_net_orig.hdf5',
                                  r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights_with_DA\dense_net_DAHV.hdf5',
                                  r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights_with_DA\dense_net_DARK.hdf5',
                                  r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights_with_DA\dense_net_BR.hdf5',
                                  r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights_with_DA\dense_net_NI.hdf5']

data_models = ['DA Brightening',
                'DA Noise Injection']

def Squeeze_models(TL_models):
    acc_loss_hist = dict()

    for model in TL_models:
        if model == 'Original dataset':
            r_SqueezeNet = TL.SqueezeN('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[0],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['Original dataset'] = r_SqueezeNet

    for model in TL_models:
        if model == 'DA H+V flipping':
            r_SqueezeNet = TL.SqueezeN('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[1],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_HV, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA H+V flipping'] = r_SqueezeNet

    for model in TL_models:
        if model == 'DA Darkening':
            r_SqueezeNet = TL.SqueezeN('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[2],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_DARK, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA Darkening'] = r_SqueezeNet

    for model in TL_models:
        if model == 'DA Brightening':
            r_SqueezeNet = TL.SqueezeN('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[3],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_BR, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA Brightening'] = r_SqueezeNet

    for model in TL_models:
        if model == 'DA Noise Injection':
            r_SqueezeNet = TL.SqueezeN('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[4],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_NI, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA Noise Injection'] = r_SqueezeNet

    return acc_loss_hist

def DenseN121_models(TL_models):
    acc_loss_hist = dict()

    for model in TL_models:
        if model == 'Original dataset':
            r_DenseN121 = TL.DenseN121('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[0],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['Original dataset'] = r_DenseN121

    for model in TL_models:
        if model == 'DA H+V flipping':
            r_DenseN121 = TL.DenseN121('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[1],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_HV, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA H+V flipping'] = r_DenseN121

    for model in TL_models:
        if model == 'DA Darkening':
            r_DenseN121 = TL.DenseN121('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[2],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_DARK, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA Darkening'] = r_DenseN121

    for model in TL_models:
        if model == 'DA Brightening':
            r_DenseN121 = TL.DenseN121('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[3],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_BR, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA Brightening'] = r_DenseN121

    for model in TL_models:
        if model == 'DA Noise Injection':
            r_DenseN121 = TL.DenseN121('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_DA[4],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train_NI, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DA Noise Injection'] = r_DenseN121

    return acc_loss_hist

acc_loss_hist = DenseN121_models(TL_models=data_models)

