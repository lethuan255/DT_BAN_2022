import numpy as np
import pandas as pd
import csv

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.merge import Concatenate

import keras
from keras import optimizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint

train_df = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\test\10setPermissionApi\train-1.csv') #đưa vao tap du lieu chung, ko can chia train va test
#train_df=train_df.sample(100)
train_data1 = np.array(train_df, dtype='u1')
test_df = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\test\10setPermissionApi\file-1.csv') #đưa vao tap du lieu chung, ko can chia train va test
#test_df=test_df.sample(100)
test_data1 = np.array(test_df, dtype='u1')
val_df = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\test\10setPermissionApi\file-2.csv') #đưa vao tap du lieu chung, ko can chia train va test
#val_df=val_df.sample(100)
val_data1 = np.array(val_df, dtype='u1')
# Đưa dữ liệu vào mảng
train_y = train_data1[1:, 0]
train_data= train_data1[1:,1:-5]

print(train_data.shape)

def Vector2CoOMatrix(vector):
    for i in range(vector.shape[0]):
        cur=np.zeros(shape=(vector.shape[0],),dtype=np.dtype('u1'))
        if vector[i]==1:
            for j in range(vector.shape[0]):
                if vector[j]==1:
                    cur[j]=1
        if i==0:
            coo_matrix=cur
        else:
            coo_matrix=np.vstack((coo_matrix,cur))
    return coo_matrix
def CoOlize(vectors):
    for i in range(vectors.shape[0]):
        if i==0:
            output=np.expand_dims(Vector2CoOMatrix(vectors[i,:]),axis=0)
        else:
            output=np.vstack((output,np.expand_dims(Vector2CoOMatrix(vectors[i,:]),axis=0)))
    return output

'''
val_y = val_data1[1:, 0]
val_data= val_data1[1:,1:-5]

test_y = test_data1[1:, 0]
test_data= test_data1[1:,1:-5]

train_x1 = train_data[:,:398]
train_x2 = train_data[:,398:598]
print("so chieu cua train1: ", train_x1.shape)
print("so chieu cua train2: ", train_x2.shape)
val_x1 = val_data[:,:398]
val_x2 = val_data[:,398:]
print("so chieu cua val1: ", val_x1.shape)
print("so chieu cua val2: ", val_x2.shape)
test_x1 = test_data[:,:398]
test_x2 = test_data[:,398:598]
print("so chieu cua test1: ", test_x1.shape)
print("so chieu cua test2: ", test_x2.shape)


train_x1=np.pad(train_x1,((0,0),(0,2)),"constant")
val_x1= np.pad(val_x1,((0,0),(0,2)),"constant")
test_x1=np.pad(test_x1,((0,0),(0,2)),"constant")

train_x2=np.pad(train_x2,((0,0),(0,25)),"constant")
val_x2= np.pad(val_x2,((0,0),(0,25)),"constant")
test_x2=np.pad(test_x2,((0,0),(0,25)),"constant")
'''
train_x1 = train_data[:,:398]
train_x2 = train_data[:,398:598]
print("so chieu cua train1: ", train_x1.shape)
print("so chieu cua train2: ", train_x2.shape)
concat_test_val = np.concatenate((test_data1,val_data1))
test_y = concat_test_val[1:, 0]
test_data= concat_test_val[1:,1:-5]

test_x1 = test_data[:,:398]
test_x2 = test_data[:,398:598]
print("so chieu cua test1: ", test_x1.shape)
print("so chieu cua test2: ", test_x2.shape)

'''
train_x1=np.pad(train_x1,((0,0),(0,2)),"constant")
test_x1=np.pad(test_x1,((0,0),(0,2)),"constant")
print("so chieu cua train_x1: ", train_x1.shape)
train_x2=np.pad(train_x2,((0,0),(0,25)),"constant")
test_x2=np.pad(test_x2,((0,0),(0,25)),"constant")
'''
# Khởi tạo hằng số
BATCH_SIZE = 64     #số lượng các ví dụ huấn luyện trong 1 đợt. Kích thước càng lớn thì càng cần bộ nhớ nhiều.
IMG_SIZE1 = 398
IMG_SIZE2 = 200

N_CLASSES = 180
LR = 0.001
N_EPOCHS = 10


train_x1=CoOlize(train_x1)
train_x2=CoOlize(train_x2)
test_x1=CoOlize(test_x1)
test_x2=CoOlize(test_x2)

#print("chay xong đồng hiện. train_x1: ",train_x1)




#ĐƯa dữ liệu về dạng phù hợp
train_x1 = train_x1.reshape(-1, IMG_SIZE1, IMG_SIZE1, 1)
#val_x1 = val_x1.reshape(-1, IMG_SIZE1, IMG_SIZE1, 1)
test_x1 = test_x1.reshape(-1, IMG_SIZE1, IMG_SIZE1, 1)
print("reshape trainx1: ", train_x1.shape)
print("reshape test_x1: ", test_x1.shape)
train_x2 = train_x2.reshape(-1, IMG_SIZE2, IMG_SIZE2, 1)
#val_x2 = val_x2.reshape(-1, IMG_SIZE2, IMG_SIZE2, 1)
test_x2 = test_x2.reshape(-1, IMG_SIZE2, IMG_SIZE2, 1)
print("reshape trainx2: ", train_x2.shape)
print("reshape test_x2: ", test_x2.shape)
# convert class vectors to binary class matrices
train_y = keras.utils.to_categorical(train_y, N_CLASSES)
test_y = keras.utils.to_categorical(test_y, N_CLASSES)
#val_y = keras.utils.to_categorical(val_y, N_CLASSES)

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(IMG_SIZE1, IMG_SIZE1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(N_CLASSES, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_data, train_y,
          batch_size=BATCH_SIZE,
          epochs=N_EPOCHS,
          verbose=1,
          validation_data=(test_data, test_y))

'''
print("start")
def create_convolution_layers(input_img):
	model = Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE1, IMG_SIZE1, 1))(input_img)
	model = LeakyReLU(alpha=0.1)(model)
	model = MaxPooling2D((2, 2),padding='same')(model)
	model = Dropout(0.25)(model)
	model = Conv2D(32, (3, 3), padding='same')(model)
	model = LeakyReLU(alpha=0.1)(model)
	model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
	model = Dropout(0.25)(model)
	model = Conv2D(32, (3, 3), padding='same')(model)
	model = LeakyReLU(alpha=0.1)(model)
	model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
	model = Dropout(0.25)(model)
	model = Conv2D(64, (3, 3), padding='same')(model)
	model = LeakyReLU(alpha=0.1)(model)
	model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
	model = Dropout(0.25)(model)
	model = Conv2D(64, (3, 3), padding='same')(model)
	model = LeakyReLU(alpha=0.1)(model)
	model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
	model = Dropout(0.25)(model)
	model = Conv2D(64, (3, 3), padding='same')(model)
	model = LeakyReLU(alpha=0.1)(model)
	model = MaxPooling2D(pool_size=(2, 2),padding='same')(model)
	model = Dropout(0.25)(model)
	model = Flatten()(model)
	return model



input1 = Input(shape=(IMG_SIZE1, IMG_SIZE1, 1))
current_model = create_convolution_layers(input1)

input2 = Input(shape=(IMG_SIZE2, IMG_SIZE2, 1))
voltage_model = create_convolution_layers(input2)

#conv = concatenate([current_model, voltage_model])

#conv = Flatten()(conv)
conv = Concatenate()([current_model,voltage_model])
dense = Dense(1024)(conv)
dense = LeakyReLU(alpha=0.1)(dense)
dense = Dropout(0.5)(dense)

output = Dense(N_CLASSES, activation='softmax')(dense)

model = Model(inputs=[input1, input2], outputs=[output])

opt = optimizers.Adam()
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])


print(model.summary())
best_weights_file="weights.best.hdf5"
checkpoint = ModelCheckpoint(best_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks = [checkpoint]
model.fit([train_x1, train_x2], train_y,batch_size=BATCH_SIZE,epochs=N_EPOCHS, verbose=1,validation_data=([test_x1, test_x2], test_y))

#model.load_weights(best_weights_file)

final_loss, final_acc = model.evaluate([test_x1, test_x2], test_y, verbose=1)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))  
'''
input1 = Conv2D(32, kernel_size=3, activation='relu',input_shape=(IMG_SIZE1, IMG_SIZE1, 1))(train_x1)
input1 = MaxPooling2D(pool_size=(2, 2))(input1)
input1 = Conv2D(64, kernel_size=3, activation='relu')(input1)
input1 = MaxPooling2D(pool_size=(2, 2))(input1)

print("qua input1")

input2 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(IMG_SIZE2, IMG_SIZE2, 1))(train_x2)
input2 = MaxPooling2D(pool_size=(2, 2))(input2)
input2 = Conv2D(64, (3, 3), activation='relu')(input2)
input2 = MaxPooling2D(pool_size=(2, 2))(input2)


input1 = Flatten()(input1)
input2 = Flatten()(input2)



merge = Concatenate()([model2,model2])
print("merger: ",merge)
output = Dense(1024, activation='sigmoid')(merge)
output= Dense(N_CLASSES, activation='softmax')(output)
model = Model(inputs=visible, outputs=output)

print(model.summary())
print("chay đến đây.")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit([train_x1,train_x2], train_y,
          batch_size=BATCH_SIZE,
          epochs=N_EPOCHS,
          verbose=1,
          validation_data=([test_x2,test_x2], test_y))



score = model.evaluate([test_x2,test_x2], test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
