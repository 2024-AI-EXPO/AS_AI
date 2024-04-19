import numpy as np
import os
import cv2
from keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# C:/Users/modeep1/Downloads/archive/asl_alphabet_train/asl_alphabet_train
# C:/Users/modeep1/Downloads/archive/asl_alphabet_test/asl_alphabet_test
train_dir = '/home/modeep3/바탕화면/AI-testv_1/asl_alphabet_train/asl_alphabet_train'
test_dir = '/home/modeep3/바탕화면/AI-testv_1/asl_alphabet_test/asl_alphabet_test'
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
               'M': 12,
               'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
               'Y': 24,
               'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
size = (64, 64)
init = Orthogonal(gain=1.0, seed=None)


def load_train_data():
    train_y = []
    train_x = []
    print("LOADING DATA FROM : ", end="")
    for folder in os.listdir(train_dir):
        print(folder, end=' | ')
        for image in os.listdir(train_dir + "/" + folder):
            # read image
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image, 0)
            # resize image
            temp_img = cv2.resize(temp_img, size)
            # load converted classes
            train_y.append(labels_dict[folder])
            train_x.append(temp_img)
    # convert X_train to numpy
    train_x = np.array(train_x)
    # normalize pixels of X_train
    train_x = train_x.astype('float32')/255.0
    # convert from 1-channel to 3-channel
    train_x = np.stack((train_x,)*3, axis=-1)
    # convert Y_train to numpy
    train_y = np.array(train_y)

    return train_x, train_y


def load_test_data():
    labels = []
    test_x = []
    for image in os.listdir(test_dir):
        # read image
        temp_img = cv2.imread(test_dir + '/' + image, 0)
        # resize image
        temp_img = cv2.resize(temp_img, size)
        # load converted classes
        labels.append(labels_dict[image.split('_')[0]])
        test_x.append(temp_img)
    # convert X_test to numpy
    test_x = np.array(test_x)
    # normalize pixels of X_test
    test_x = test_x.astype('float32')/255.0
    # convert from 1-channel to 3-channel in Gray
    test_x = np.stack((test_x,)*3, axis=-1)
    # convert Y_test to numpy
    test_y = np.array(labels)

    return test_x, test_y


X_train, Y_train = load_train_data()
X_test, Y_test = load_test_data()

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
model.add(Dense(29, activation='softmax', kernel_initializer=init))

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
        ModelCheckpoint('model64.keras', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=10, mode='auto'),
    ]
)
