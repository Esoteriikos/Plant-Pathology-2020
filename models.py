import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Flatten, Dropout, MaxPooling2D


def model_efn():
    input_shape = (150, 150, 3)
    classes = 4
    
    model = efn.EfficientNetB7(weights='imagenet', input_shape=input_shape, pooling='max', include_top=False)
    x = model.output
    output = Dense(classes, activation='softmax')(x)

    return Model(inputs=model.input, outputs=output)


def model_v3():
    input_shape = (150, 150, 3)
    classes = 4
    
    model = Sequential([
            Conv2D(8, (5, 5), padding="same", activation="relu", input_shape=input_shape),
            BatchNormalization(),
            #150 150 8
          
            Conv2D(16, (5, 5), padding="same", activation="relu"),
            Dropout(0.2),

            Conv2D(32, (7, 7), activation="relu"),
            # 144 144 32
            MaxPooling2D(2,2),
            # 72 72 32
            
            Conv2D(64, (5, 5), padding="same", activation="relu"),
            Dropout(0.25),
            Conv2D(64, (5, 5), padding="valid", activation="relu"),
            # 68 68 64
            MaxPooling2D(2,2),
            # 34 34 64
            BatchNormalization(),
            #tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01),

            
            Conv2D(128, (5, 5), padding="same", activation="relu"),
         #   Dropout(0.25),
            Conv2D(128, (5, 5), padding="valid", activation="relu"),
           # 30 30 128
            MaxPooling2D(3, 3),
            # 10 10 128

            Conv2D(256, (5, 5), padding="same", activation="relu"),
            Dropout(0.3),
            Conv2D(256, (5, 5), padding="valid", activation="relu"),
            # 6 6 256
            MaxPooling2D(2,2),
            # 3 3 256
            BatchNormalization(),


            Flatten(),
            #Dense(4096, activation="relu"),
            #Dense(2048, activation="relu"),
            #Dropout(0.25),
            Dense(1024, activation="relu"),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            Dense(64, activation="relu"),
            Dense(classes, activation="softmax")
            
            ])
    print(model.summary())
    return model

'''
if __name__=="__main__":
    #m1 = model_v1()
    #print(m1.summary())
    m2 = model_v2()
    print(m2.summary())
'''
