import datetime
import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.imagenet_utils import obtain_input_shape
from tensorflow.keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils.layer_utils import get_source_inputs
from tensorflow.keras.utils import get_file
from keras.utils import layer_utils
import os



def MNetV2(train_layers, class_weights, save_model_path, save_ckp_path, image_height, image_width, ds_train, ds_val, lr_rate, num_epochs):

    num_classes = 4
    epochs = num_epochs

    if train_layers == 'scratch':
        mobile_model = MobileNetV2(input_shape=(image_height, image_width, 3), include_top=False, weights=None)
        mobile_model.trainable = True
    if train_layers == 'TL_classifier':
        mobile_model = MobileNetV2(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        mobile_model.trainable = False
    if train_layers == 'TL_all':
        mobile_model = MobileNetV2(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        mobile_model.trainable = True

    model_MobileV2 = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        mobile_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
 
    model_MobileV2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    
    print ("Save checkpoint file in: " + save_ckp_path)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= save_ckp_path)
    
    mcp_save = ModelCheckpoint(save_model_path,
                                save_best_only=True, monitor='val_accuracy', mode='auto')

    # list together
    callbacks = [mcp_save, tensorboard_callback]

    r_mobileV2 = model_MobileV2.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights)

    return r_mobileV2

def IncV3(train_layers, class_weights, save_model_path, save_ckp_path, image_height, image_width, ds_train, ds_val, lr_rate, num_epochs):
    num_classes = 4
    epochs = num_epochs
 
    if train_layers == 'scratch':
        InceptionV3_model = InceptionV3(input_shape=(image_height, image_width, 3), include_top=False, weights=None)
        InceptionV3_model.trainable = True
    if train_layers == 'TL_classifier':
        InceptionV3_model = InceptionV3(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        InceptionV3_model.trainable = False
    if train_layers == 'TL_all':
        InceptionV3_model = InceptionV3(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        InceptionV3_model.trainable = True

    model_InceptionV3 = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        InceptionV3_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])


    print ("Save checkpoint file in: " + save_ckp_path)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= save_ckp_path)
    
    mcp_save = ModelCheckpoint(save_model_path,
                                save_best_only=True, monitor='val_accuracy', mode='auto')

    # list together
    callbacks = [mcp_save, tensorboard_callback]


    r_InceptionV3 = model_InceptionV3.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights)

    return r_InceptionV3

def ResN50(train_layers, class_weights, save_model_path, save_ckp_path, image_height, image_width, ds_train, ds_val, lr_rate, num_epochs):
    num_classes = 4
    epochs = num_epochs

    if train_layers == 'scratch':
        ResNet50_model = ResNet50(input_shape=(image_height, image_width, 3), include_top=False, weights=None)
        ResNet50_model.trainable = True
    if train_layers == 'TL_classifier':
        ResNet50_model = ResNet50(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        ResNet50_model.trainable = False
    if train_layers == 'TL_all':
        ResNet50_model = ResNet50(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        ResNet50_model.trainable = True

    model_ResNet50 = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        ResNet50_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

   
    model_ResNet50.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    print ("Save checkpoint file in: " + save_ckp_path)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= save_ckp_path)
    
    mcp_save = ModelCheckpoint(save_model_path,
                                save_best_only=True, monitor='val_accuracy', mode='auto')

    # list together
    callbacks = [mcp_save, tensorboard_callback]

    r_ResNet50 = model_ResNet50.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights)

    return r_ResNet50

def DenseN121(train_layers, class_weights, save_model_path, save_ckp_path, image_height, image_width, ds_train, ds_val, lr_rate, num_epochs):
    num_classes = 4
    epochs = num_epochs

    if train_layers == 'scratch':
        DenseNet121_model = DenseNet121(input_shape=(image_height, image_width, 3), include_top=False, weights=None)
        DenseNet121_model.trainable = True
    if train_layers == 'TL_classifier':
        DenseNet121_model = DenseNet121(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        DenseNet121_model.trainable = False
    if train_layers == 'TL_all':
        DenseNet121_model = DenseNet121(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        DenseNet121_model.trainable = True

    model_DenseNet121 = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        DenseNet121_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    print ("Save checkpoint file in: " + save_ckp_path)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= save_ckp_path)
    
    mcp_save = ModelCheckpoint(save_model_path,
                                save_best_only=True, monitor='val_accuracy', mode='auto')

    # list together
    callbacks = [mcp_save, tensorboard_callback]

    r_DenseNet121 = model_DenseNet121.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights)

    return r_DenseNet121

def SqueezeN(train_layers, class_weights, save_model_path, save_ckp_path, image_height, image_width, ds_train, ds_val, lr_rate, num_epochs):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
    WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

    # Modular function for Fire Node

    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x = Activation('relu', name=s_id + relu + sq1x1)(x)

        left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x

    # Original SqueezeNet from paper.

    def SqueezeNet(include_top=True, weights='imagenet',
                   input_tensor=None, input_shape=None,
                   pooling=None,
                   classes=1000):
        """Instantiates the SqueezeNet architecture.
        """

        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        if weights == 'imagenet' and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')

        input_shape = obtain_input_shape(input_shape,
                                          default_size=227,
                                          min_size=48,
                                          data_format=K.image_data_format(),
                                          require_flatten=include_top)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)

        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        if include_top:
            # It's not obvious where to cut the network...
            # Could do the 8th or 9th layer... some work recommends cutting earlier layers.

            x = Dropout(0.5, name='drop9')(x)

            x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
            x = Activation('relu', name='relu_conv10')(x)
            x = GlobalAveragePooling2D()(x)
            x = Activation('softmax', name='loss')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)
            elif pooling == None:
                pass
            else:
                raise ValueError("Unknown argument for 'pooling'=" + pooling)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        model = Model(inputs, x, name='squeezenet')

        # load weights
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')

            model.load_weights(weights_path)
            if K.backend() == 'theano':
                layer_utils.convert_all_kernels_in_model(model)

        return model

    num_classes = 4
    epochs = num_epochs


    if train_layers == 'scratch':
        SqueezeNet_model = SqueezeNet(input_shape=(image_height, image_width, 3), include_top=False, weights=None)
        SqueezeNet_model.trainable = True
    if train_layers == 'TL_classifier':
        SqueezeNet_model = SqueezeNet(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        SqueezeNet_model.trainable = False
    if train_layers == 'TL_all':
        SqueezeNet_model = SqueezeNet(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
        SqueezeNet_model.trainable = True
        
    model_SqueezeNet = keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
        SqueezeNet_model,
        keras.layers.Conv2D(1000, (1, 1), strides=1, padding='valid', name='conv10'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

  
    model_SqueezeNet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    print ("Save checkpoint file in: " + save_ckp_path)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= save_ckp_path)
    
    mcp_save = ModelCheckpoint(save_model_path,
                                save_best_only=True, monitor='val_accuracy', mode='auto')

    # list together
    callbacks = [mcp_save, tensorboard_callback]

    r_SqueezeNet = model_SqueezeNet.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights)

    return r_SqueezeNet

