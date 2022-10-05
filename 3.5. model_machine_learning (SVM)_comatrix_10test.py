#Thư viện
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn.ensemble as ske
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


#Algorithm comparison
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
#SVM
from sklearn.svm import SVC

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
# Đưa file csv vào để train và test.
#data = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\machine learning\total_Permission_API1.csv')

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# tree base feature selection
# Feature selection using Trees Classifier

fsel = ske.ExtraTreesClassifier(random_state=0).fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]
#Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_new, y ,test_size=0.2, random_state=42)


features = []

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
print('-->> {nb_features} features identified as important:')
for f in range(nb_features):
    print(f"{f + 1}. feature {data.columns[2+indices[f]]}")

#take care of the feature order
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])

print(f'Tong so feature sau khi toi uu: {X_new.shape[1]}')
print(f'so feature loai bo: {X.shape[1]} - {X_new.shape[1]} = {X.shape[1] - X_new.shape[1]}')

#chạy dataset thuần mà không tối giản
#X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=0)
'''
#lấy dữ liệu cho đồng bộ với CNN. Train là 80% còn test + val => test (20%)
train_df = pd.read_csv(r'D:\Thay Thuan\10setPermissionApi\train-7.csv') #đưa vao tap du lieu chung, ko can chia train va test

#train_df=train_df.sample(100)
train_data1 = np.array(train_df, dtype='u1')
#train_data1[np.lexsort(np.fliplr(train_data1).T)]
test_df = pd.read_csv(r'D:\Thay Thuan\10setPermissionApi\file-7.csv') #đưa vao tap du lieu chung, ko can chia train va test
#test_df=test_df.sample(2)
test_data = np.array(test_df, dtype='u1')
val_df = pd.read_csv(r'D:\Thay Thuan\10setPermissionApi\file-8.csv') #đưa vao tap du lieu chung, ko can chia train va test
#val_df=val_df.sample(2)
val_data = np.array(val_df, dtype='u1')


train_label = train_data1[1:, 0]
train_data = train_data1[1:, 1:-5]
X_train = train_data       #từ row 0  tới x
y_train = train_label
print("my train X: ", X_train)
print("my train Y: ", y_train)

concat_test_val = np.concatenate((test_data,val_data))

print("test_data",test_data.shape)
print("val_data",val_data.shape)
print("my test: ",concat_test_val.shape)
y_test = concat_test_val[1:, 0]
X_test = concat_test_val[1:, 1:-5]

test_x1,test_x9, test_y1, test_y9 = train_test_split(X_test, y_test, test_size=0.2)            #test_x9 chiem 20%; test_x1 chiem 80%
test_x10,test_x9, test_y10, test_y9 = train_test_split(test_x9, test_y9, test_size=0.5)

test_x5,test_x1, test_y5, test_y1 = train_test_split(test_x1, test_y1, test_size=0.5)
test_x1,test_x3, test_y1, test_y3 = train_test_split(test_x1, test_y1, test_size=0.5)
test_x1,test_x2, test_y1, test_y2 = train_test_split(test_x1, test_y1, test_size=0.5)

test_x4,test_x3, test_y4, test_y3 = train_test_split(test_x3, test_y3, test_size=0.5)
test_x5,test_x7, test_y5, test_y7 = train_test_split(test_x5, test_y5, test_size=0.5)
test_x6,test_x5, test_y6, test_y5 = train_test_split(test_x5, test_y5, test_size=0.5)
test_x8,test_x7, test_y8, test_y7 = train_test_split(test_x7, test_y7, test_size=0.5)
print("y test: ", y_test)

X_train=CoOlize(X_train)
X_test=CoOlize(X_test)

test_x1=CoOlize(test_x1)
test_x2=CoOlize(test_x2)
test_x3=CoOlize(test_x3)
test_x4=CoOlize(test_x4)
test_x5=CoOlize(test_x5)

test_x6=CoOlize(test_x6)
test_x7=CoOlize(test_x7)
test_x8=CoOlize(test_x8)
test_x9=CoOlize(test_x9)
test_x10=CoOlize(test_x10)

IMG_SIZE1 = 598
N_CLASSES = 180
#X_train = X_train.reshape(IMG_SIZE1, IMG_SIZE1)
#X_test = X_test.reshape(-1, IMG_SIZE1, IMG_SIZE1, 1)
#y_train = keras.utils.to_categorical(y_train, N_CLASSES)
#y_test = keras.utils.to_categorical(y_test, N_CLASSES)

print("x_train: ", X_train)
#print("Làm phẳng: ", X_train.transpose(2,0,1).reshape(len(X_train),-1))
print("nhãn: ", y_train)
#hết đoạn lấy dữ liệu theo CNN

algorithms = {
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=50, random_state=0),
        "RandomForest": ske.RandomForestClassifier(n_estimators=100, random_state=0),
        #"GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50, random_state=0),
        #"AdaBoost": ske.AdaBoostClassifier(n_estimators=100, random_state=0),
        #"GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        #"SVM": SVC(kernel='rbf', random_state=0)     #thêm SVM vào
    }

results = {}

print("\nNow testing algorithms")

#Testing algo
#train = X_train.transpose(2,0,1).reshape(len(X_train),-1)
#test = X_test.transpose(2,0,1).reshape(len(X_test),-1)
train = X_train.reshape(len(X_train),IMG_SIZE1*IMG_SIZE1)
#test = X_test.reshape(len(X_test),IMG_SIZE1*IMG_SIZE1)
test1 = test_x1.reshape(len(test_x1),IMG_SIZE1*IMG_SIZE1)
test2 = test_x2.reshape(len(test_x2),IMG_SIZE1*IMG_SIZE1)
test3 = test_x3.reshape(len(test_x3),IMG_SIZE1*IMG_SIZE1)
test4 = test_x4.reshape(len(test_x4),IMG_SIZE1*IMG_SIZE1)
test5 = test_x5.reshape(len(test_x5),IMG_SIZE1*IMG_SIZE1)
test6 = test_x6.reshape(len(test_x6),IMG_SIZE1*IMG_SIZE1)
test7 = test_x7.reshape(len(test_x7),IMG_SIZE1*IMG_SIZE1)
test8 = test_x8.reshape(len(test_x8),IMG_SIZE1*IMG_SIZE1)
test9 = test_x9.reshape(len(test_x9),IMG_SIZE1*IMG_SIZE1)
test10 = test_x10.reshape(len(test_x10),IMG_SIZE1*IMG_SIZE1)



for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(train, y_train)
    score1 = clf.score(test1, test_y1)
    score2 = clf.score(test2, test_y2)
    score3 = clf.score(test3, test_y3)
    score4 = clf.score(test4, test_y4)
    score5 = clf.score(test5, test_y5)
    score6 = clf.score(test6, test_y6)
    score7 = clf.score(test7, test_y7)
    score8 = clf.score(test8, test_y8)
    score9 = clf.score(test9, test_y9)
    score10 = clf.score(test10, test_y10)
    score = (score1+score2+score3+score4+score5+score6+score7+score8+score9+score10)/10
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

winner = max(results, key=results.get)
print(f'\nWinner algorithm is {winner} with a {results[winner]*100} %% success')


# Identify false and true positive rates
##clf_winner = algorithms[winner]
##y_pred_winner = clf_winner.predict(test)
##cm_winner = confusion_matrix(y_test, y_pred_winner)
#print(f"False positive rate : {((cm_winner[0][1] / float(sum(cm_winner[0])))*100)}")
#print(f'False negative rate : {((cm_winner[1][0] / float(sum(cm_winner[1]))*100))}')
#=====================================

#svm
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(train, y_train)
print("train xong SVM.")
# Predicting the Test set results
#y_pred = classifier.predict(test)

# Making the Confusion Matrix

#cm = confusion_matrix(y_test, y_pred)

# score
ketqua1 = classifier.score(test1, test_y1)
ketqua2 = classifier.score(test2, test_y2)
ketqua3 = classifier.score(test3, test_y3)
ketqua4 = classifier.score(test4, test_y4)
ketqua5 = classifier.score(test5, test_y5)
ketqua6 = classifier.score(test6, test_y6)
ketqua7 = classifier.score(test7, test_y7)
ketqua8 = classifier.score(test8, test_y8)
ketqua9 = classifier.score(test9, test_y9)
ketqua10 = classifier.score(test10, test_y10)
ketqua = (ketqua1+ketqua2+ketqua3+ketqua4+ketqua5+ketqua6+ketqua7+ketqua8+ketqua9+ketqua10)/10
#print("SVM:    ",classifier.score(X_test, y_test)) 
#print("SVM : %f %%" % (cm*100))
print("phân lớp bằng SVM: ",ketqua*100)



