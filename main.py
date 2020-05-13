import os
import time
import models
import datetime
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from load_process_data import data_from_id
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, TensorBoard, Callback, ModelCheckpoint

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

nh = nw = 150
nc = 3

num_classes = 4
classes = {1:"Healthy", 2:"Multiple_diseases", 3:"Rust", 4:"Scab"}

epochs = 100
batch_size = 6

ACCURACY_THRESHOLD = 0.999
VAL_ACCURACY_THRESHOLD = 0.95

learning_rate = 0.00015
beta_1 = 0.9
beta_2 = 0.999

reduce_lr_factor = 0.35
patience = 8

images_path = r'plant-pathology-2020-fgvc7\images\\'
train_path = r"plant-pathology-2020-fgvc7\train.csv"
test_path = r"plant-pathology-2020-fgvc7\test.csv"

print('Loading Training data')
train = data_from_id(images_path=images_path, csv_path=train_path, img_shape=(nw, nh), target=True)
x_train, y_train = train.image_target_array(image_label='image_id', target_labels=['healthy', 'multiple_diseases', 'rust', 'scab'])

#print("Loading Test data")
#test = data_from_id(images_path=images_path, img_shape=(256,256), csv_path=test_path)
#x_test = test.image_target_array(image_label='image_id', target_labels=['healthy', 'multiple_diseases', 'rust', 'scab'])

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
#print(" x_train = {} \n y_train = {} \n x_val = {} \n y_val = {}\n".format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))

#os.mkdir(r"trained_model\v3-3")

class Callbacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%3==0:
            model.save_weights(r"trained_model\v2-1\modelv2-1-{}.h5".format(epoch))
        if logs['acc'] > ACCURACY_THRESHOLD:# and logs['val_acc']> VAL_ACCURACY_THRESHOLD:
            self.model.stop_training = True

tensor_board = TensorBoard(log_dir=r"logs\model-{}".format(int(time.time())))
csv_logger = CSVLogger(filename=r"CSVLogger\model-logs-{}".format(int(time.time())))

checkpoint_filepath = r'tmp\v2-1\checkpoint.h5'

model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                            save_weights_only=True,
                                            monitor='val_acc',
                                            mode='max',
                                            save_best_only=True)
                                            
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=patience, min_lr=0.00001)

train_datagen = ImageDataGenerator(rotation_range=30,
                                   rescale=(1./255),
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.25,
                                   zoom_range=0.25,
                                   horizontal_flip=True
                                   )
                                   
train_datagen.fit(x_train)

#val_datagen = ImageDataGenerator(rescale=1./255)
#val_datagen.fit(x_val)

model = models.model_efn()
#print(model.summary)

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2),
              metrics=['acc']
              )


history = model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=len(x_train) / batch_size,
                              epochs=epochs,
                              verbose=1,
                             # validation_data=val_datagen.flow(x_val,y_val, batch_size=batch_size),
                             # validation_steps=len(x_val) / batch_size,
                              callbacks=[tensor_board,
                                         csv_logger,
                                         Callbacks()
                                         model_checkpoint_callback,
                  #                       reduce_lr
                                         ]
                              )
                              
model.save_weights('model_vefn.h5')
