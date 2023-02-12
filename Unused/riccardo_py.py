# imports
import utils.train_val_test_dataset_import as tvt
import utils.class_imbalances as ci
import models.TL_models as TL
import tensorflow as tf

# constants
image_height=224
image_width=224
batch_size=16
nr_epochs=50

# paths
# path_train=[r'PATH_DATASET_1',r'PATH DATASET_2']
path_train=[r'C:\Users\rtaormina\Dropbox\Werk\RESEARCH\Andre_s_paper\plastic_datasets\train_27m_0deg_center_bank_aug_resized']
path_val=r'C:\Users\rtaormina\Dropbox\Werk\RESEARCH\Andre_s_paper\plastic_datasets\test_27m_0deg_center_bank_VAL'

# load datasets
ds_train_1, ds_val_1 = tvt.import_dataset_train_val(path_train[0], path_val, image_height, image_width, batch_size)
# ds_train_2, ds_val_2 = tvt.import_dataset_train_val(path_train[1], path_val, image_height, image_width, batch_size)

# autotune
AUTOTUNE = tf.data.AUTOTUNE
ds_train_1 = ds_train_1.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
ds_val_1 = ds_val_1.cache().prefetch(buffer_size=AUTOTUNE)

# ds_train_2 = ds_train_2.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# ds_val_2 = ds_val_2.cache().prefetch(buffer_size=AUTOTUNE)

# class weights
class_weights_train = ci.class_weights(path_train[0])

# paths to model save
save_model = [r'C:\Users\rtaormina\Dropbox\Werk\RESEARCH\Andre_s_paper\code\model_save\squeeze_net_1.hdf5'
                r'C:\Users\rtaormina\Dropbox\Werk\RESEARCH\Andre_s_paper\code\model_save\dense_net_1.hdf5']
#                r'C:\Users\Andre\Desktop\Research paper\test\squeeze_net_2.hdf5',
#                r'C:\Users\Andre\Desktop\Research paper\test\dense_net_2.hdf5']

# train models
r_squeeze_1 = TL.SqueezeN('Yes', class_weights=class_weights_train, save_model_path=save_model[0], 
    image_height=image_height, image_width=image_width, 
    ds_train=ds_train_1, ds_val=ds_val_1, lr_rate=0.0001, nr_epochs=nr_epochs)

r_dense_1 = TL.DenseN121('Yes', class_weights=class_weights_train, save_model_path=save_model[1], 
    image_height=image_height, image_width=image_width, 
    ds_train=ds_train_1, ds_val=ds_val_1, lr_rate=0.0001, nr_epochs=nr_epochs)

# r_squeeze_2 = TL.SqueezeN('Yes', class_weights=class_weights_train, save_model_path=save_model[2], 
    # image_height=image_height, image_width=image_width, 
    # ds_train=ds_train_2, ds_val=ds_val_2, lr_rate=0.0001, nr_epochs=nr_epochs)

# r_dense_2 = TL.DenseN121('Yes', class_weights=class_weights_train, save_model_path=save_model[3], 
    # image_height=image_height, image_width=image_width, 
    # ds_train=ds_train_2, ds_val=ds_val_2, lr_rate=0.0001, nr_epochs=nr_epochs)