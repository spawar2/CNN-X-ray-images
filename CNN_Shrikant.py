# Shrikant Pawar, CNN on chest X-ray, 01/22/2020
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.utils import shuffle
df = pd.read_csv('/Users/yalegenomecenter/Desktop/NIH-Machine-Learning-Example/test/Test_Sample.csv')
diseases = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia']
for disease in diseases :
    df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
df['Age']=df['Patient Age'].apply(lambda x: x[:-1]).astype(int)
df['Age Type']=df['Patient Age'].apply(lambda x: x[-1:])
df.loc[df['Age Type']=='M',['Age']] = df[df['Age Type']=='M']['Age'].apply(lambda x: round(x/12.)).astype(int)
df.loc[df['Age Type']=='D',['Age']] = df[df['Age Type']=='D']['Age'].apply(lambda x: round(x/365.)).astype(int)
df = df.drop(df['Age'].sort_values(ascending=False).head(16).index)
df['Age'] = df['Age']/df['Age'].max()
df = df.join(pd.get_dummies(df['Patient Gender']))
df = df.join(pd.get_dummies(df['View Position']))
df = shuffle(df)
data = df[['Age', 'F', 'M', 'AP', 'PA']]
data = np.array(data)
labels = df[diseases].as_matrix()
files_list = ('/Users/yalegenomecenter/Desktop/NIH-Machine-Learning-Example/images/' + df['Image Index']).tolist()
labelB = (df[diseases].sum(axis=1)>0).tolist()
labelB = np.array(labelB, dtype=int)
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path, shape):
    img = image.load_img(img_path, target_size=shape)
    x = image.img_to_array(img)/255
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, shape):
    list_of_tensors = [path_to_tensor(img_path, shape) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

train_labels = labelB[:30][:, np.newaxis]
valid_labels = labelB[30:70][:, np.newaxis]
test_labels = labelB[70:][:, np.newaxis]

train_data = data[:30]
valid_data = data[30:70]
test_data = data[70:]

img_shape = (64, 64)
train_tensors = paths_to_tensor(files_list[:30], shape = img_shape)
valid_tensors = paths_to_tensor(files_list[30:70], shape = img_shape)
test_tensors = paths_to_tensor(files_list[70:], shape = img_shape)



import time

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import regularizers, initializers, optimizers

model = Sequential()

model.add(Conv2D(filters=16, 
                 kernel_size=7,
                 padding='same', 
                 activation='relu', 
                 input_shape=train_tensors.shape[1:]))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, 
                 kernel_size=5,
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, 
                 kernel_size=5,
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, 
                 kernel_size=5,
                 strides=2,
                 padding='same', 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


import time

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras import regularizers, applications, optimizers, initializers
from keras.preprocessing.image import ImageDataGenerator
import ssl

from keras import backend as K

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

def precision_threshold(threshold = 0.5):
    def precision(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(y_pred)
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

def fbeta_score_threshold(beta = 1, threshold = 0.5):
    def fbeta_score(y_true, y_pred):
        threshold_value = threshold
        beta_value = beta
        p = precision_threshold(threshold_value)(y_true, y_pred)
        r = recall_threshold(threshold_value)(y_true, y_pred)
        bb = beta_value ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score
    return fbeta_score

model.compile(optimizer='sgd', loss='binary_crossentropy', 
              metrics=[precision_threshold(threshold = 0.5), 
                       recall_threshold(threshold = 0.5), 
                       fbeta_score_threshold(beta=0.5, threshold = 0.5),
                      'accuracy'])

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import numpy as np

epochs = 20
batch_size = 32

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
log = CSVLogger('saved_models/log_bCNN_rgb.csv')
checkpointer = ModelCheckpoint(filepath='saved_models/bCNN.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

start = time.time()

model.fit(train_tensors, train_labels, 
          validation_data=(valid_tensors, valid_labels),
          epochs=epochs, batch_size=batch_size, callbacks=[checkpointer, log, earlystop], verbose=1)

print("training time: %.2f minutes"%((time.time()-start)/60))

model.load_weights('saved_models/bCNN.best.from_scratch.hdf5')
prediction = model.predict(test_tensors)

threshold = 0.5
beta = 0.5

pre = K.eval(precision_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
fsc = K.eval(fbeta_score_threshold(beta = beta, threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))

print ("Precision: %f %%\nRecall: %f %%\nFscore: %f %%"% (pre, rec, fsc))


K.eval(binary_accuracy(K.variable(value=test_labels),
                                   K.variable(value=prediction)))

threshold = 0.6
beta = 0.5

pre = K.eval(precision_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
rec = K.eval(recall_threshold(threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))
fsc = K.eval(fbeta_score_threshold(beta = beta, threshold = threshold)(K.variable(value=test_labels),
                                   K.variable(value=prediction)))



