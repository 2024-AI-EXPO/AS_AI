import numpy as np
import os
import cv2
from keras.layers import Flatten, Dense, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.initializers import Orthogonal
from keras.applications.vgg16 import VGG16
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras.metrics import AUC
from keras.metrics import Precision
from keras.metrics import Recall
from tensorflow.python.client import device_lib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

data_index = np.arange(1, 2001)
np.random.seed(2016)
np.random.shuffle(data_index)

data_dir = '/home/modeep3/Github/AS_AI/dataset'
labels_dict = {'A': 0, 'B': 1}
init = Orthogonal(gain=1.0, seed=None)
size = (224, 224)

def load_train_data():
    train_y = []
    train_x = []
    # print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(data_dir):
        # print(folder, end=' | ')
        for idx in data_index[:1500]:
            temp_img = cv2.imread(data_dir + '/' + folder + f'/{folder}_{idx}.jpg', 0)
            temp_img = cv2.resize(temp_img, size)
            train_y.append(labels_dict[folder])
            train_x.append(temp_img)
            
        print(f'{folder} end')

    train_x = np.array(train_x)
    train_x = train_x.astype('float32')/255.0
    train_x = np.stack((train_x,)*3, axis=-1)
    train_y = np.array(train_y)

    return train_x, train_y


def load_test_data():
    train_y = []
    train_x = []
    # print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(data_dir):
        # print(folder, end=' | ')
        for idx in data_index[1500:]:
            temp_img = cv2.imread(data_dir + '/' + folder + f'/{folder}_{idx}.jpg', 0)
            temp_img = cv2.resize(temp_img, size)
            train_y.append(labels_dict[folder])
            train_x.append(temp_img)
            
        print(f'{folder} end')

    train_x = np.array(train_x)
    train_x = train_x.astype('float32')/255.0
    train_x = np.stack((train_x,)*3, axis=-1)
    train_y = np.array(train_y)

    return train_x, train_y


X_train, Y_train = load_train_data()
X_test, Y_test = load_test_data()
print(X_train.shape, Y_train.shape)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

print('Loaded', len(Y_test), 'images for testing,', 'Test data shape =', Y_test.shape)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False
)
datagen.fit(X_train)

model = Sequential()
model.add(Input(shape=(size[0], size[1], 3)))
model.add(VGG16(weights='imagenet', include_top=False))
model.add(Flatten())
model.add(Dense(420, activation='relu', kernel_initializer=init))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax', kernel_initializer=init))

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=[
        AUC(name='Accuracy'),
        Precision(name='Precision'),
        Recall(name='Recall')
    ]
)

model.summary()

tf.config.run_functions_eagerly(True)
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=32),
    epochs=100,
    validation_data=(X_test, Y_test),
    callbacks=[
        ModelCheckpoint('/home/modeep3/Github/AS_AI/test_models/model_weight.ckpt', verbose=1, save_weights_only=True),
        ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=10, mode='auto'),
    ]
)
