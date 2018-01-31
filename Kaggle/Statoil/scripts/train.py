import models

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from subprocess import check_output
from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator

print(check_output(["ls", "./dataset/"]).decode("utf8"))

#Load data
train = pd.read_json("./dataset/train.json")
test = pd.read_json("./dataset/test.json")
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
print("loading files done!")

# Process the data and splitting
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], 
                         x_band2[:, :, :, np.newaxis], 
                        ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
X_angle_train = np.array(train.inc_angle)
y_train = np.array(train["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], 
                        x_band2[:, :, :, np.newaxis], 
                        ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
X_angle_test = np.array(test.inc_angle)

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
                    , X_angle_train, y_train, random_state=123, train_size=0.75)

print("X_train: ", X_train.shape)
print("X_angle_train: ", X_angle_train.shape)
print("y_train: ", y_train.shape)

print("***************")

print("X_valid: ", X_valid.shape)
print("X_angle_valid: ", X_angle_valid.shape)
print("y_valid: ", y_valid.shape)

print("***************")

print("X_test: ", X_test.shape)
print("X_angle_test: ", X_angle_test.shape)

model = models.get_model()
model.summary()

gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 40)

def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]
            
gen_flow = gen_flow_for_two_inputs(X_train, X_angle_train, y_train)


file_path = "model_weights.hdf5"
callbacks = models.get_callbacks(filepath=file_path, patience=5)

model.fit_generator(gen_flow
                    ,validation_data=([X_valid, X_angle_valid], y_valid)
                     ,steps_per_epoch=len(X_train)/32
                    ,epochs = 25
                     ,callbacks=modelcallbacks)


model.load_weights(filepath=file_path)

print("Train evaluate:")
print(model.evaluate([X_train, X_angle_train], y_train, verbose=1, batch_size=200))

print("####################")

print("watch list evaluate:")
print(model.evaluate([X_valid, X_angle_valid], y_valid, verbose=1, batch_size=200))

prediction = model.predict([X_test, X_angle_test], verbose=1, batch_size=200)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.head(10)

submission.to_csv("./submission.csv", index=False)
