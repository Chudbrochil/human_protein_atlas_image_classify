import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import keras
import PIL
import cv2
import tensorflow as tf
import warnings

from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
from PIL import Image
from sklearn.utils import class_weight, shuffle
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

class data_generator:

    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), 28))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image/255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image_red_ch = Image.open(path+'_red.png')
        image_yellow_ch = Image.open(path+'_yellow.png')
        image_green_ch = Image.open(path+'_green.png')
        image_blue_ch = Image.open(path+'_blue.png')
        image = np.stack((
        np.array(image_red_ch),
        np.array(image_green_ch),
        np.array(image_blue_ch)), -1)
        image = cv2.resize(image, (shape[0], shape[1]))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
        iaa.OneOf([
            iaa.Affine(rotate=0),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = input_shape)

    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)

    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(n_out, activation='sigmoid')(x)

    model = Model(input_tensor, output)

    return model

SIZE = 512

# Load dataset info
path_to_train = '../input/train/'
data = pd.read_csv('../input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

epochs = 20; batch_size = 16
checkpoint = ModelCheckpoint('../working/InceptionV3.h5', monitor='val_f1', verbose=1,save_best_only=True, mode='max', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor='val_f1', mode='max', patience=6, verbose=1)
logger = CSVLogger('training.csv')

callbacks_list = [checkpoint, early, reduceLROnPlat, logger]

# split data into train, valid
indexes = np.arange(train_dataset_info.shape[0])
np.random.shuffle(indexes)
train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)

# create train and valid datagens
train_generator = data_generator.create_train(train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=True)
validation_generator = data_generator.create_train(train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)

# warm up model
model = create_model(input_shape=(SIZE,SIZE,3), n_out=28)

# train all layers
for layer in model.layers:
    layer.trainable = True

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4),metrics=['accuracy', f1])

model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),
    validation_data=validation_generator,
    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),
    epochs=epochs,
    verbose=2,
    callbacks=callbacks_list)

# Create submit
submit = pd.read_csv('../input/sample_submission.csv')
predicted = []
draw_predict = []
model.load_weights('../working/InceptionV3.h5')
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = data_generator.load_image(path, (SIZE,SIZE,3))/255.
    score_predict = model.predict(image[np.newaxis])[0]
    draw_predict.append(score_predict)
    label_predict = np.arange(28)[score_predict>=0.2] # This is a magic number...
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
np.save('draw_predict_InceptionV3.npy', score_predict)
submit.to_csv('submit_InceptionV3.csv', index=False)
