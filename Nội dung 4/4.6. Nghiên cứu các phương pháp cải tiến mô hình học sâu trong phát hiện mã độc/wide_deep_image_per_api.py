#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Khai báo những thư viện cần thiết
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Embedding, Dense, Flatten, Activation, Dropout, concatenate
from keras.layers.advanced_activations import ReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import to_categorical, plot_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import time

print("Bắt đầu.")
# Đường dẫn đến 2 file train và test
TRAIN_FILE_PATH = r'/content/drive/My Drive/Colab Notebooks/10set (đủ mã)/AMD_Benign_ver2(tong hop)/train-0.csv'

TEST_FILE_PATH = r'/content/drive/My Drive/Colab Notebooks/10set (đủ mã)/AMD_Benign_ver2(tong hop)/file-0.csv'
VAL_FILE_PATH = r'/content/drive/My Drive/Colab Notebooks/10set (đủ mã)/AMD_Benign_ver2(tong hop)/file-1.csv'
#FILE_TOTAL_PATH = r'F:\Thay Thuan\data\image128_per_api_drebin_benign\Drebin_Benign_128(Img_Per_API)_ver3.csv'
#tt_data = pd.read_csv(FILE_TOTAL_PATH, header=None, skiprows = 1)            #Test thử cùng với row 48 để chạy đủ file total
# Khai báo những tham số cần thiết
SIZE = 128
N_CLASSES = 228
N_EPOCHS = 50
N_BATCH_SIZE = 32
N_PERMISSION_COLUMNS = 877     #drebin 829  AMD 879    và all 908
N_API_COLUMNS = 1000
N_IMAGE_COLUMNS = 16384

# Đọc dữ liệu train và test
train_data = pd.read_csv(TRAIN_FILE_PATH, header=None, skiprows = 1)
#train_data=train_data.sample(100) 
train_data.dropna(how='any', axis=0)
test_data = pd.read_csv(TEST_FILE_PATH, header=None, skiprows = 1)
#test_data=test_data.sample(100)
test_data.dropna(how='any', axis=0)
val_data = pd.read_csv(VAL_FILE_PATH,header=None, skiprows = 1)
#val_data=val_data.sample(100)
val_data.dropna(how='any', axis=0)

# #test_data = np.concatenate((test_data,val_data)) 
# train_data = pd.concat([train_data, val_data])
# #train_data = pd.concat([train_data, test_data])
# # Tổng số cột của một dòng
# cols = train_data.columns.values
# #train_data=tt_data.sample(frac = 0.9) 
# #test_data=tt_data.sample(frac = 0.1)  #Test thử cùng với row 20,21 để chạy đủ file total
# print("ab=",test_data.shape)
# # Cột đầu tiên là label, tên cột là '2'.
# # Những cột còn lại đóng vai trò là feature
# LABEL_COLUMN = cols[2]
# FEATURE_COLUMNS = cols[3:]
# print("feature_columns = ", FEATURE_COLUMNS.shape)         #Tính tổng số đặc trưng trong file csv
# # Số lượng cột Permission feature được lấy từ cột thứ 0 đến cột N_PERMISSION_COLUMNS-1
# # Số lượng cột API feature được lấy từ cột thứ N_PERMISSION_COLUMNS đến hết.
# #TH1: sử dụng cho file có các giá trị ảnh
# IMAGE_COLUMNS = FEATURE_COLUMNS[0: N_IMAGE_COLUMNS]
# PERMISSION_COLUMNS = FEATURE_COLUMNS[N_IMAGE_COLUMNS: N_IMAGE_COLUMNS+N_PERMISSION_COLUMNS]
# API_COLUMNS = FEATURE_COLUMNS[N_IMAGE_COLUMNS+N_PERMISSION_COLUMNS:]
# print('PERMISSION_COLUMNS = ', PERMISSION_COLUMNS)
# print('API_COLUMNS = ', API_COLUMNS)
# print('IMAGE_COLUMNS = ', IMAGE_COLUMNS)
# '''
# #TH2: Sử dụng cho các file không có giá trị ảnh
# #IMAGE_COLUMNS = FEATURE_COLUMNS[0: N_IMAGE_COLUMNS]
# PERMISSION_COLUMNS = FEATURE_COLUMNS[:N_PERMISSION_COLUMNS]
# API_COLUMNS = FEATURE_COLUMNS[N_PERMISSION_COLUMNS:N_PERMISSION_COLUMNS+1833]

# print("PERMISSION_COLUMNS: ",PERMISSION_COLUMNS.shape)
# print("API_COLUMNS: ",API_COLUMNS.shape)
# # In ra thông tin các cột đã lấy
# #print('PERMISSION_COLUMNS = ', PERMISSION_COLUMNS)
# #print('API_COLUMNS = ', API_COLUMNS)
# '''



# # Hàm này để xử lý data trước khi thực hiện các bước sau
# def preprocessing():
#     all_data = pd.concat([train_data, test_data])
#     # Cột đầu tiên (cột số 2) được định nghĩa là label
#     all_data[LABEL_COLUMN] = all_data[LABEL_COLUMN].astype(int)
#     y = all_data[LABEL_COLUMN].values
#     print("y:",y)
#     all_data.pop(0) # remove file name
#     all_data.pop(1) # remove  label name
#     all_data.pop(LABEL_COLUMN) # remove label id
#     print("gia tri lon nhan cua nhan:", max(y))

#     # Chia data train và data test từ data đã đưa vào
#     train_size = len(train_data)
#     x_train = all_data.iloc[:train_size]
#     y_train = y[:train_size]
#     x_test = all_data.iloc[train_size:]
#     y_test = y[train_size:]

#     print(x_train.shape)

#     # Chia tập x_train, x_test thành 2 tập con là per và api
#     x_train_per = np.array(x_train[PERMISSION_COLUMNS])
#     x_train_api = np.array(x_train[API_COLUMNS])
#     x_test_per = np.array(x_test[PERMISSION_COLUMNS])
#     x_test_api = np.array(x_test[API_COLUMNS])
#     print("train per:", x_train_per.shape)
#     #Đầu vào dành cho wide
#     x_train_per_w = np.array(x_train[N_IMAGE_COLUMNS])
#     x_train_api_w = np.array(x_train[PERMISSION_COLUMNS+API_COLUMNS])
#     x_test_per_w = np.array(x_test[N_IMAGE_COLUMNS])
#     x_test_api_w = np.array(x_test[PERMISSION_COLUMNS+API_COLUMNS])
#     print("train per for wide:", x_train_per_w.shape)

#     return (x_train, y_train, x_test, y_test, x_train_per, x_test_per, x_train_api, x_test_api, x_train_per_w, x_test_per_w, x_train_api_w, x_test_api_w, all_data)


# def plot_graphs(history):
#     plt.plot(history.history['accuracy'])
#     plt.xlabel("Epochs")
#     plt.ylabel('accuracy')
#     plt.legend(['accuracy'])
#     plt.savefig('plot_graphs.png')
#     plt.close()


# class Wide_and_Deep:
#     # Định nghĩa class Wide_and_Deep và khai báo, khởi tạo những property cần thiết
#     def __init__(self, mode='wide and deep'):
#         self.mode = mode

#         x_train, y_train, x_test, y_test, x_train_per, x_test_per, x_train_api, x_test_api, x_train_per_w, x_test_per_w, x_train_api_w, x_test_api_w, all_data = preprocessing()
#         self.x_train = x_train
#         self.y_train = y_train
#         self.x_test = x_test
#         self.y_test = y_test
#         self.x_train_per = x_train_per
#         self.x_test_per = x_test_per
#         self.x_train_api = x_train_api
#         self.x_test_api = x_test_api
#         self.x_train_per_w = x_train_per_w
#         self.x_test_per_w = x_test_per_w
#         self.x_train_api_w = x_train_api_w
#         self.x_test_api_w = x_test_api_w
#         self.all_data = all_data
#         self.poly = PolynomialFeatures(degree=2, interaction_only=True)

#         '''self.x_train_per_poly = self.poly.fit_transform(x_train_per)
#         self.x_test_per_poly = self.poly.transform(x_test_per)
#         self.x_train_api_poly = self.poly.fit_transform(x_train_api)
#         self.x_test_api_poly = self.poly.transform(x_test_api)
# '''
#         self.per_input = None
#         self.api_input = None
#         self.deep_component_outlayer = None
#         self.logistic_input = None
#         self.logistic_per_input = None
#         self.logistic_api_input = None
#         # self.logistic_api_input = None
#         self.model = None
#         self.history = None

#     # Hàm này để tạo deep model
#     def deep_component(self):
#         per_inputs = []
#         api_inputs = []
#         per_embeds = []
#         api_embeds = []
#         dims_per = []
#         dims_api = []

#         for i in range(len(PERMISSION_COLUMNS)):
#             input_i = Input(shape=(1,), dtype='int32')
#             dim = len(np.unique(self.all_data[PERMISSION_COLUMNS[i]]))
#             embed_dim = int(np.ceil(dim ** 0.25))
#             dims_per.append(embed_dim)
#             embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
#             flatten_i = Flatten()(embed_i)
#             per_inputs.append(input_i)
#             per_embeds.append(flatten_i)

#         for i in range(len(API_COLUMNS)):
#             input_i = Input(shape=(1,), dtype='int32')
#             dim = len(np.unique(self.all_data[API_COLUMNS[i]]))
#             embed_dim = int(np.ceil(dim ** 0.25))
#             dims_api.append(embed_dim)
#             embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
#             flatten_i = Flatten()(embed_i)
#             api_inputs.append(input_i)
#             api_embeds.append(flatten_i)

#         print(per_embeds)
#         print(api_embeds)
#         print(dims_per)
#         print(dims_api)

#         # Đưa input per vào model
#         deep_per_input = Input(shape=(N_PERMISSION_COLUMNS,))
#         deep_per_dense = Dense(128, use_bias=False)(deep_per_input)

#         # Đưa input api vào model
#         deep_api_input = Input(shape=(N_API_COLUMNS,))
#         deep_api_dense = Dense(128, use_bias=False)(deep_api_input)

#         # Sau khi các input đã đi qua Full connected thì merge lại với nhau
#         # concat_embeds = concatenate(per_embeds + api_embeds)
#         concat_embeds = concatenate([deep_per_dense] + [deep_api_dense])
#         concat_embeds = Activation('relu')(concat_embeds)
#         bn_concat = BatchNormalization()(concat_embeds)

#         # Tiến hành cho qua những lớp ẩn
#         fc1 = Dense(2048, use_bias=False)(bn_concat)
#         ac1 = ReLU()(fc1)
#         bn1 = BatchNormalization()(ac1)
#         fc2 = Dense(1024, use_bias=False)(bn1)
#         ac2 = ReLU()(fc2)
#         bn2 = BatchNormalization()(ac2)
#         fc3 = Dense(512, use_bias=False)(bn2)
#         ac3 = ReLU()(fc3)
#         bn3 = BatchNormalization()(ac3)
#         fc4 = Dense(256)(bn3)
#         dropout = Dropout(0.25)
#         ac4 = ReLU()(fc4)

#         self.per_input = deep_per_input
#         self.api_input = deep_api_input

#         # Đầu ra cuối cùng của deep model
#         self.deep_component_outlayer = ac4
#         print('Finish create deep model')

#     # Hàm này để tạo wide model
#     def wide_model(self):
#         # Đưa input per vào model
#         dim_per = self.x_train_per_w.shape[1]
#         print("x_train_per_w = ", self.x_train_per_w.shape)
#         print("x_train_per = ", self.x_train_per.shape)
#         self.logistic_per_input = Input(shape=(dim_per,))
#         print("logistic_per_input = ",self.logistic_per_input.shape)
#         # Đưa input api vào model
#         dim_api = self.x_train_api_w.shape[1]
#         print("x_train_api_w = ", self.x_train_api_w.shape)
#         self.logistic_api_input = Input(shape=(dim_api,))
#         print("logistic_api_input = ",self.logistic_api_input.shape)

#     # Hàm này thực hiện merge wide & deep model lại với nhau
#     def create_model(self):
#         print('Starting create model ...')
#         self.deep_component()
#         self.wide_model()

#         if self.mode == 'wide and deep':
#             # Output layer của model cần phải có các thông tin output của từng model đã tạo
#             out_layer = concatenate([self.deep_component_outlayer, self.logistic_per_input, self.logistic_api_input])
#             # Inputs cũng cần phải được đưa vào những thông tin đúng thứ tự đã định nghĩa của từng model
#             # Nên xem summary để biết thứ tự đưa vào các input hoặc xem thứ tự gọi các lệnh input của từng model
#             inputs = [self.per_input] + [self.api_input] + [self.logistic_per_input] + [self.logistic_api_input]
#         elif self.mode == 'deep':
#             out_layer = self.deep_component_outlayer
#             inputs = [self.per_input] + [self.api_input]
#         else:
#             print('wrong mode')
#             return

#         # Output cuối cùng của model
#         output = Dense(N_CLASSES, activation='softmax')(out_layer)
#         print(f'input = {len(inputs)}')

#         # Tạo model với thông tin inputs và output đã định nghĩa
#         self.model = Model(inputs=inputs, outputs=output)
#         print(self.model.summary())

#     # Hàm này dùng để train model, ứng thứ tự các input đã khai báo, cần đưa vào model đúng thứ tự tập input train
#     def train_model(self):
#         if not self.model:
#             print('You have to create model first')
#             return

#         if self.mode == 'wide and deep':
#             print("chạy vào train wide and deep")
#             input_data = [self.x_train_per] + [self.x_train_api] + [self.x_train_per_w] + [self.x_train_api_w]
#             print("xxxxxxxx")
#         elif self.mode == 'deep':
#             input_data = [self.x_train_per] + [self.x_train_api]
#         else:
#             print('wrong mode')
#             return

#         # Do bài toán ở đây là multi classification nên loss phải là 'categorical_crossentropy'
#         self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         print("qua compile")
#         # Chuyển đối số chiều của tập y_train từ [len(y_train), None] về [len(y_train), N_CLASSES]
#         self.y_train = to_categorical(self.y_train, N_CLASSES)
#         # Định nghĩa callbacks theo 'accuracy'
#         es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max')
#         # callbacks = [es]
#         callbacks = []
#         # Trong quá trình train có thể sử dụng callbacks hoặc không.
#         self.history = self.model.fit(input_data, self.y_train, epochs=N_EPOCHS, batch_size=N_BATCH_SIZE)
#     # Hàm này dùng để đánh giá kết quả model đã train bằng tập test
#     # Các tham số đưa vào được tính toán tương tự với khi train
#     def evaluate_model(self):
#         if not self.model:
#             print('You have to create model first')
#             return

#         if self.mode == 'wide and deep':
#             print("chạy vào test wide and deep")
#             input_data = [self.x_test_per] + [self.x_test_api] + [self.x_test_per_w] + [self.x_test_api_w]
#         elif self.mode == 'deep':
#             input_data = [self.x_test_per] + [self.x_test_api]

#         else:
#             print('wrong mode')
#             return

#         self.y_test = to_categorical(self.y_test,N_CLASSES)
#         loss, acc = self.model.evaluate(input_data, self.y_test)
#         print(f'\ntest_loss: {loss} - test_acc: {acc}')

#     # Lưu lại model dưới dạng .h5 (keras output format) để sử dụng về sau
#     def save_model(self, filename='wide_and_deep.h5'):
#         self.model.save(filename)


# if __name__ == '__main__':
#     # Khai báo 1 đối tượng Wide_and_Deep() và sử dụng các hàm đã định nghĩa của Wide_and_Deep
#     t0 = time.time()
#     print("bắt đầu chạy winde and deep")
#     wide_deep_net = Wide_and_Deep()
#     t1 = time.time()
#     print("chạy xong wide and deep")
#     print("bắt đầu chạy create model")
#     wide_deep_net.create_model()
#     print("chạy xong create model")
#     print("bắt đầu chạy train model")
#     wide_deep_net.train_model()
#     print("chạy xong train model")
#     print("bắt đầu chạy evaluate_model")
#     wide_deep_net.evaluate_model()
#     print("chạy xong evaluate_model")
#     wide_deep_net.save_model()

#     #plot_graphs(wide_deep_net.history)

#     # Lưu lại thông tin của model dứa dạng file .png
#     #plot_model(wide_deep_net.model, to_file='model.png', show_shapes=True, show_layer_names=False)
#     t2 = time.time()

#     print("thoi gian chạy model không tính comatrix: %.2f phút." %((t2-t1)/60))
#     print("thoi gian chạy xong model: %.2f phút." %((t2-t0)/60))
#     print('Hoàn thành việc huấn luyện và kiểm tra!')
    


# In[ ]:


#test_data = np.concatenate((test_data,val_data)) 
train_data = pd.concat([train_data, val_data])
#train_data = pd.concat([train_data, test_data])
# Tổng số cột của một dòng
cols = train_data.columns.values
#train_data=tt_data.sample(frac = 0.9) 
#test_data=tt_data.sample(frac = 0.1)  #Test thử cùng với row 20,21 để chạy đủ file total
print("ab=",test_data.shape)
# Cột đầu tiên là label, tên cột là '2'.
# Những cột còn lại đóng vai trò là feature
LABEL_COLUMN = cols[2]
FEATURE_COLUMNS = cols[3:]
print("feature_columns = ", FEATURE_COLUMNS.shape)         #Tính tổng số đặc trưng trong file csv
# Số lượng cột Permission feature được lấy từ cột thứ 0 đến cột N_PERMISSION_COLUMNS-1
# Số lượng cột API feature được lấy từ cột thứ N_PERMISSION_COLUMNS đến hết.
#TH1: sử dụng cho file có các giá trị ảnh
IMAGE_COLUMNS = FEATURE_COLUMNS[0: N_IMAGE_COLUMNS]
PERMISSION_COLUMNS = FEATURE_COLUMNS[N_IMAGE_COLUMNS: ]
API_COLUMNS = FEATURE_COLUMNS[N_IMAGE_COLUMNS+N_PERMISSION_COLUMNS:]
print('PERMISSION_COLUMNS = ', PERMISSION_COLUMNS)
print('API_COLUMNS = ', API_COLUMNS)
print('IMAGE_COLUMNS = ', IMAGE_COLUMNS)
'''
#TH2: Sử dụng cho các file không có giá trị ảnh
#IMAGE_COLUMNS = FEATURE_COLUMNS[0: N_IMAGE_COLUMNS]
PERMISSION_COLUMNS = FEATURE_COLUMNS[:N_PERMISSION_COLUMNS]
API_COLUMNS = FEATURE_COLUMNS[N_PERMISSION_COLUMNS:N_PERMISSION_COLUMNS+1833]

print("PERMISSION_COLUMNS: ",PERMISSION_COLUMNS.shape)
print("API_COLUMNS: ",API_COLUMNS.shape)
# In ra thông tin các cột đã lấy
#print('PERMISSION_COLUMNS = ', PERMISSION_COLUMNS)
#print('API_COLUMNS = ', API_COLUMNS)
'''



# Hàm này để xử lý data trước khi thực hiện các bước sau
def preprocessing():
    all_data = pd.concat([train_data, test_data])
    # Cột đầu tiên (cột số 2) được định nghĩa là label
    all_data[LABEL_COLUMN] = all_data[LABEL_COLUMN].astype(int)
    y = all_data[LABEL_COLUMN].values
    print("y:",y)
    all_data.pop(0) # remove file name
    all_data.pop(1) # remove  label name
    all_data.pop(LABEL_COLUMN) # remove label id
    print("gia tri lon nhan cua nhan:", max(y))

    # Chia data train và data test từ data đã đưa vào
    train_size = len(train_data)
    x_train = all_data.iloc[:train_size]
    y_train = y[:train_size]
    x_test = all_data.iloc[train_size:]
    y_test = y[train_size:]

    print(x_train.shape)

    # Chia tập x_train, x_test thành 2 tập con là per và api
    x_train_per = np.array(x_train[PERMISSION_COLUMNS])
    x_train_api = np.array(x_train[API_COLUMNS])
    x_test_per = np.array(x_test[PERMISSION_COLUMNS])
    x_test_api = np.array(x_test[API_COLUMNS])
    print("train per:", x_train_per.shape)
    #Đầu vào dành cho wide
    x_train_per_w = np.array(x_train[N_IMAGE_COLUMNS])
    x_train_api_w = np.array(x_train[PERMISSION_COLUMNS.tolist()+API_COLUMNS.tolist()])
    x_test_per_w = np.array(x_test[N_IMAGE_COLUMNS])
    x_test_api_w = np.array(x_test[PERMISSION_COLUMNS.tolist()+API_COLUMNS.tolist()])
    print("train per for wide:", x_train_per_w.shape)

    return (x_train, y_train, x_test, y_test, x_train_per, x_test_per, x_train_api, x_test_api, x_train_per_w, x_test_per_w, x_train_api_w, x_test_api_w, all_data)


def plot_graphs(history):
    plt.plot(history.history['accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel('accuracy')
    plt.legend(['accuracy'])
    plt.savefig('plot_graphs.png')
    plt.close()


class Wide_and_Deep:
    # Định nghĩa class Wide_and_Deep và khai báo, khởi tạo những property cần thiết
    def __init__(self, mode='wide and deep'):
        self.mode = mode

        x_train, y_train, x_test, y_test, x_train_per, x_test_per, x_train_api, x_test_api, x_train_per_w, x_test_per_w, x_train_api_w, x_test_api_w, all_data = preprocessing()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_per = x_train_per
        self.x_test_per = x_test_per
        self.x_train_api = x_train_api
        self.x_test_api = x_test_api
        self.x_train_per_w = x_train_per_w
        self.x_test_per_w = x_test_per_w
        self.x_train_api_w = x_train_api_w
        self.x_test_api_w = x_test_api_w
        self.all_data = all_data
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)

        '''self.x_train_per_poly = self.poly.fit_transform(x_train_per)
        self.x_test_per_poly = self.poly.transform(x_test_per)
        self.x_train_api_poly = self.poly.fit_transform(x_train_api)
        self.x_test_api_poly = self.poly.transform(x_test_api)
'''
        self.per_input = None
        self.api_input = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.logistic_per_input = None
        self.logistic_api_input = None
        # self.logistic_api_input = None
        self.model = None
        self.history = None

    # Hàm này để tạo deep model
    def deep_component(self):
        per_inputs = []
        api_inputs = []
        per_embeds = []
        api_embeds = []
        dims_per = []
        dims_api = []

        for i in range(len(PERMISSION_COLUMNS)):
            input_i = Input(shape=(1,), dtype='int32')
            dim = len(np.unique(self.all_data[PERMISSION_COLUMNS[i]]))
            embed_dim = int(np.ceil(dim ** 0.25))
            dims_per.append(embed_dim)
            embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            per_inputs.append(input_i)
            per_embeds.append(flatten_i)

        for i in range(len(API_COLUMNS)):
            input_i = Input(shape=(1,), dtype='int32')
            dim = len(np.unique(self.all_data[API_COLUMNS[i]]))
            embed_dim = int(np.ceil(dim ** 0.25))
            dims_api.append(embed_dim)
            embed_i = Embedding(dim, embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            api_inputs.append(input_i)
            api_embeds.append(flatten_i)

        print(per_embeds)
        print(api_embeds)
        print(dims_per)
        print(dims_api)

        # Đưa input per vào model
        deep_per_input = Input(shape=(N_PERMISSION_COLUMNS,))
        deep_per_dense = Dense(128, use_bias=False)(deep_per_input)

        # Đưa input api vào model
        deep_api_input = Input(shape=(N_API_COLUMNS,))
        deep_api_dense = Dense(128, use_bias=False)(deep_api_input)

        # Sau khi các input đã đi qua Full connected thì merge lại với nhau
        # concat_embeds = concatenate(per_embeds + api_embeds)
        concat_embeds = concatenate([deep_per_dense] + [deep_api_dense])
        concat_embeds = Activation('relu')(concat_embeds)
        bn_concat = BatchNormalization()(concat_embeds)

        # Tiến hành cho qua những lớp ẩn
        fc1 = Dense(2048, use_bias=False)(bn_concat)
        ac1 = ReLU()(fc1)
        bn1 = BatchNormalization()(ac1)
        fc2 = Dense(1024, use_bias=False)(bn1)
        ac2 = ReLU()(fc2)
        bn2 = BatchNormalization()(ac2)
        fc3 = Dense(512, use_bias=False)(bn2)
        ac3 = ReLU()(fc3)
        bn3 = BatchNormalization()(ac3)
        fc4 = Dense(256)(bn3)
        dropout = Dropout(0.25)
        ac4 = ReLU()(fc4)

        self.per_input = deep_per_input
        self.api_input = deep_api_input

        # Đầu ra cuối cùng của deep model
        self.deep_component_outlayer = ac4
        print('Finish create deep model')

    # Hàm này để tạo wide model
    def wide_model(self):
        # Đưa input per vào model
        dim_per = self.x_train_per_w.shape
        print("x_train_per_w = ", self.x_train_per_w.shape)
        print("x_train_per = ", self.x_train_per.shape)
        self.logistic_per_input = Input(shape=(dim_per,))
        print("logistic_per_input = ",self.logistic_per_input.shape)
        # Đưa input api vào model
        dim_api = self.x_train_api_w.shape[1]
        print("x_train_api_w = ", self.x_train_api_w.shape)
        self.logistic_api_input = Input(shape=(dim_api,))
        print("logistic_api_input = ",self.logistic_api_input.shape)

    # Hàm này thực hiện merge wide & deep model lại với nhau
    def create_model(self):
        print('Starting create model ...')
        self.deep_component()
        self.wide_model()

        if self.mode == 'wide and deep':
            # Output layer của model cần phải có các thông tin output của từng model đã tạo
            out_layer = concatenate([self.deep_component_outlayer, self.logistic_per_input, self.logistic_api_input])
            # Inputs cũng cần phải được đưa vào những thông tin đúng thứ tự đã định nghĩa của từng model
            # Nên xem summary để biết thứ tự đưa vào các input hoặc xem thứ tự gọi các lệnh input của từng model
            inputs = [self.per_input] + [self.api_input] + [self.logistic_per_input] + [self.logistic_api_input]
        elif self.mode == 'deep':
            out_layer = self.deep_component_outlayer
            inputs = [self.per_input] + [self.api_input]
        else:
            print('wrong mode')
            return

        # Output cuối cùng của model
        output = Dense(N_CLASSES, activation='softmax')(out_layer)
        print(f'input = {len(inputs)}')

        # Tạo model với thông tin inputs và output đã định nghĩa
        self.model = Model(inputs=inputs, outputs=output)
        print(self.model.summary())

    # Hàm này dùng để train model, ứng thứ tự các input đã khai báo, cần đưa vào model đúng thứ tự tập input train
    def train_model(self):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            print("chạy vào train wide and deep")
            input_data = [self.x_train_per] + [self.x_train_api] + [self.x_train_per_w] + [self.x_train_api_w]
            print("xxxxxxxx")
        elif self.mode == 'deep':
            input_data = [self.x_train_per] + [self.x_train_api]
        else:
            print('wrong mode')
            return

        # Do bài toán ở đây là multi classification nên loss phải là 'categorical_crossentropy'
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("qua compile")
        # Chuyển đối số chiều của tập y_train từ [len(y_train), None] về [len(y_train), N_CLASSES]
        self.y_train = to_categorical(self.y_train, N_CLASSES)
        # Định nghĩa callbacks theo 'accuracy'
        es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='max')
        # callbacks = [es]
        callbacks = []
        # Trong quá trình train có thể sử dụng callbacks hoặc không.
        self.history = self.model.fit(input_data, self.y_train, epochs=N_EPOCHS, batch_size=N_BATCH_SIZE)
    # Hàm này dùng để đánh giá kết quả model đã train bằng tập test
    # Các tham số đưa vào được tính toán tương tự với khi train
    def evaluate_model(self):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            print("chạy vào test wide and deep")
            input_data = [self.x_test_per] + [self.x_test_api] + [self.x_test_per_w] + [self.x_test_api_w]
        elif self.mode == 'deep':
            input_data = [self.x_test_per] + [self.x_test_api]

        else:
            print('wrong mode')
            return

        self.y_test = to_categorical(self.y_test,N_CLASSES)
        loss, acc = self.model.evaluate(input_data, self.y_test)
        print(f'\ntest_loss: {loss} - test_acc: {acc}')

    # Lưu lại model dưới dạng .h5 (keras output format) để sử dụng về sau
    def save_model(self, filename='wide_and_deep.h5'):
        self.model.save(filename)


if __name__ == '__main__':
    # Khai báo 1 đối tượng Wide_and_Deep() và sử dụng các hàm đã định nghĩa của Wide_and_Deep
    t0 = time.time()
    print("bắt đầu chạy winde and deep")
    wide_deep_net = Wide_and_Deep()
    t1 = time.time()
    print("chạy xong wide and deep")
    print("bắt đầu chạy create model")
    wide_deep_net.create_model()
    print("chạy xong create model")
    print("bắt đầu chạy train model")
    wide_deep_net.train_model()
    print("chạy xong train model")
    print("bắt đầu chạy evaluate_model")
    wide_deep_net.evaluate_model()
    print("chạy xong evaluate_model")
    wide_deep_net.save_model()

    #plot_graphs(wide_deep_net.history)

    # Lưu lại thông tin của model dứa dạng file .png
    #plot_model(wide_deep_net.model, to_file='model.png', show_shapes=True, show_layer_names=False)
    t2 = time.time()

    print("thoi gian chạy model không tính comatrix: %.2f phút." %((t2-t1)/60))
    print("thoi gian chạy xong model: %.2f phút." %((t2-t0)/60))
    print('Hoàn thành việc huấn luyện và kiểm tra!')

