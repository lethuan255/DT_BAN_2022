#Thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
train_df = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\test\10setPermissionApi\train-9.csv') #đưa vao tap du lieu chung, ko can chia train va test
#train_df=train_df.sample(100)
train_data = np.array(train_df, dtype='u1')
test_df = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\test\10setPermissionApi\file-9.csv') #đưa vao tap du lieu chung, ko can chia train va test
#test_df=test_df.sample(100)
test_data = np.array(test_df, dtype='u1')
val_df = pd.read_csv(r'D:\Phát hiện mã độc bằng học sâu\phân tích code CNN\test\10setPermissionApi\file-0.csv') #đưa vao tap du lieu chung, ko can chia train va test
#val_df=val_df.sample(100)
val_data = np.array(val_df, dtype='u1')


train_label = train_data[1:, 0]
train_data = train_data[1:, 1:-5]
X_train = train_data       #từ row 0  tới x
y_train = train_label
print("my train X: ", X_train.shape)
print("my train Y: ", y_train.shape)

concat_test_val = np.concatenate((test_data,val_data))

print("test_data",test_data.shape)
print("val_data",val_data.shape)
print("my test: ",concat_test_val.shape)
y_test = concat_test_val[1:, 0]
X_test = concat_test_val[1:, 1:-5]
#hết đoạn lấy dữ liệu theo CNN

algorithms = {
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10, random_state=0),
        "RandomForest": ske.RandomForestClassifier(n_estimators=50, random_state=0),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50, random_state=0),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100, random_state=0),
        "GNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        #"SVM": SVC(kernel='rbf', random_state=0)     #thêm SVM vào
    }

results = {}

print("\nNow testing algorithms")

#Testing algo

for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

winner = max(results, key=results.get)
print(f'\nWinner algorithm is {winner} with a {results[winner]*100} %% success')


# Identify false and true positive rates
clf_winner = algorithms[winner]
y_pred_winner = clf_winner.predict(X_test)
cm_winner = confusion_matrix(y_test, y_pred_winner)
#print(f"False positive rate : {((cm_winner[0][1] / float(sum(cm_winner[0])))*100)}")
#print(f'False negative rate : {((cm_winner[1][0] / float(sum(cm_winner[1]))*100))}')
#=====================================

#svm
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
print("train xong SVM.")
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

# score
ketqua = classifier.score(X_test, y_test)
#print("SVM:    ",classifier.score(X_test, y_test)) 
#print("SVM : %f %%" % (cm*100))
print("phân lớp bằng SVM: ",ketqua*100)



