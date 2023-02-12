import utils.train_val_test_dataset_import as tvt
import utils.class_imbalances as ci
import models.TL_models as TL
import tensorflow as tf

image_height=224
image_width=224
batch_size=16
val_split=0.1249
test_split=0.436
nr_epochs=50
path_train=r'C:\Users\Andre\Desktop\Data MSc thesis\Training data\train_all_heights_angles'
path_val=r'C:\Users\Andre\Desktop\Data MSc thesis\Test data\test_27m_0deg_center_bank_VAL'
path_test=r'C:\Users\Andre\Desktop\Data MSc thesis\Test data\test_27m_0deg_center_bank_TEST'

ds_train, ds_val = tvt.import_dataset_train_val(path_train, path_val, image_height, image_width, batch_size)
#ds_test = tvt.import_dataset_test(path_test, image_height, image_width, batch_size)

AUTOTUNE = tf.data.AUTOTUNE
ds_train = ds_train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
ds_val = ds_val.cache().prefetch(buffer_size=AUTOTUNE)

class_weights_train = ci.class_weights(path_train)

save_model_path_pretrained = [r'C:\Users\Andre\Desktop\Research paper\pre_trained_weights\mobile_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\pre_trained_weights\squeeze_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\pre_trained_weights\resnet50_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\pre_trained_weights\inceptionv3_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\pre_trained_weights\densenet121.hdf5']

save_model_path_fromscratch = [r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\mobile_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\squeeze_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\resnet50_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\inceptionv3_net.hdf5',
                            r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\densenet121.hdf5']

save_model_path_fromscratch_ALL_HEIGHTS = [r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\squeezenet_net_ALL_HEIGHTS.hdf5',
                                            r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\densenet_net_ALL_HEIGHTS.hdf5']

def pre_trained_models(TL_models):
    acc_loss_hist = dict()

    for model in TL_models:
        if model == 'MobileNetV2':
            print(f'{model} model is running ...')
            r_mobileV2 = TL.MNetV2('No', class_weights=class_weights_train,
                                   save_model_path=save_model_path_pretrained[0],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['MobileNetV2'] = r_mobileV2

    for model in TL_models:
        if model == 'SqueezeNet':
            print(f'{model} model is running ...')
            r_SqueezeNet = TL.SqueezeN('No', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_ALL_HEIGHTS[0],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['SqueezeNet'] = r_SqueezeNet

    for model in TL_models:
        if model == 'ResNet50':
            print(f'{model} model is running ...')
            r_ResNet50 = TL.ResN50('No', class_weights=class_weights_train,
                                   save_model_path=save_model_path_pretrained[2],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['ResNet50'] = r_ResNet50

    for model in TL_models:
        if model == 'InceptionV3':
            print(f'{model} model is running ...')
            r_InceptionV3 = TL.IncV3('No', class_weights=class_weights_train,
                                   save_model_path=save_model_path_pretrained[3],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['InceptionV3'] = r_InceptionV3

    for model in TL_models:
        if model == 'DenseNet121':
            print(f'{model} model is running ...')
            r_DenseNet121 = TL.DenseN121('No', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_ALL_HEIGHTS[1],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DenseNet121'] = r_DenseNet121

    return acc_loss_hist

def from_scratch_models(TL_models):
    acc_loss_hist = dict()

    for model in TL_models:
        if model == 'MobileNetV2':
            r_mobileV2 = TL.MNetV2('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch[0],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['MobileNetV2'] = r_mobileV2

    for model in TL_models:
        if model == 'SqueezeNet':
            r_SqueezeNet = TL.SqueezeN('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_ALL_HEIGHTS[0],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['SqueezeNet'] = r_SqueezeNet

    for model in TL_models:
        if model == 'ResNet50':
            r_ResNet50 = TL.ResN50('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch[2],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['ResNet50'] = r_ResNet50

    for model in TL_models:
        if model == 'InceptionV3':
            r_InceptionV3 = TL.IncV3('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch[3],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.001, nr_epochs=nr_epochs)
            acc_loss_hist['InceptionV3'] = r_InceptionV3

    for model in TL_models:
        if model == 'DenseNet121':
            r_DenseNet121 = TL.DenseN121('Yes', class_weights=class_weights_train,
                                   save_model_path=save_model_path_fromscratch_ALL_HEIGHTS[1],
                                   image_height=image_height, image_width=image_width, ds_train=ds_train, ds_val=ds_val,
                                   lr_rate=0.0001, nr_epochs=nr_epochs)
            acc_loss_hist['DenseNet121'] = r_DenseNet121

    return acc_loss_hist


#acc_loss_hist = pre_trained_models(['MobileNetV2'])

#acc_loss_hist = pre_trained_models(['MobileNetV2', 'SqueezeNet', 'ResNet50', 'InceptionV3', 'DenseNet121'])
acc_loss_hist = from_scratch_models(['SqueezeNet', 'DenseNet121'])

#PLT.plot_results_acc(acc_loss_hist, 'Training and validation accuracy for models with pre-trained weights', 'Train acc', 'Val acc', ['b', 'g', 'r', 'c', 'm'], r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\graphs\plot_accuracy_from_scratch.png')
#PLT.plot_results_loss(acc_loss_hist, 'Training and validation loss for models with pre-trained weights', 'Train loss', 'Val loss', ['lightseagreen', 'deeppink', 'olive', 'orange', 'lime'], r'C:\Users\Andre\Desktop\Research paper\from_scratch_weights\graphs\plot_loss_from_scratch.png')
