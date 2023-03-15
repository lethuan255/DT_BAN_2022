import pandas as pd
from math import log2
import csv


df = pd.read_csv('total_per_api_amd_benign.csv',low_memory=False,header=None)
del df[0]
del df[1]

def entropyL(n):
    return -n*log2(n)

def entropy1(n,a,b):
    tp=[]   # list tp để chứa các phần tử cột a
    for i in range(1, a.__len__() + 1):
        tp.append(int(a[i]))
    tp2=[]  #list tp2 chứa các phần tử cột 0 ứng với các phần tử cột a có giá trị n
    for i in range(tp.__len__()):
        if tp[i] == n:
            tp2.append(b[i])
    t = set(tp2) #loại bỏ các phần tử trùng nhau
    e=0
    for i in t:
        e += entropyL(tp2.count(int(i))/tp2.__len__())
    return e

def entropy2(a,b):
    a.pop(0)
    tp=[]
    for i in range(1, len(a)+1):
        tp.append(int(a[i]))
    return tp.count(0)/tp.__len__()*entropy1(0,a,b)+tp.count(1)/tp.__len__()*entropy1(1,a,b)

tp2 = []
df[2].pop(0)
print("So dong: ", df[2].__len__())
for i in range(1,df[2].__len__()+1):
    tp2.append(int(df[2][i]))
tp = set(df[2])  #loại bỏ các phần tử trùng nhau

eLabel = 0
for i in tp:
    #print(int(i)," ",tp2.count(int(i)))
    eLabel += entropyL(tp2.count(int(i)) / df[2].__len__())

list = ['','',eLabel] #chứa thông tin IG của mỗi đặc trưng
print("Entropy Label: ",eLabel)

dem=0 #đếm số cột trong file sau khi xóa 2 cột đầu
for i in df:
    dem+=1

for i in range(3,dem+2):
    list.append(eLabel-entropy2(df[int(i)],tp2))

#print(list.__len__())
#lưu file thành 1 list
list2=[]
with open('total_per_api_amd_benign.csv', 'r') as rfile:
    reader = csv.reader(rfile)
    for i in reader:
        list2.append(i)

'''soHang = list2.__len__()
soCot = list2[1].__len__()
print("So hang: ",soHang)
print("So cot: ",soCot)
#print(list)
for i in range(1,soHang):     #Các giá trị 1 trong cột thay bằng entropy của cột đó
    for j in range(3,soCot):
        if int(list2[i][j]) == 1:
            list2[i][j] = list[j]'''
#print(list2)
list2.insert(1,list)  #chèn thêm hàng entropy vào hàng thứ 2
with open('total.csv', 'w',newline='') as wfile:
    writer = csv.writer(wfile)
    writer.writerows(list2)

rfile.close()
wfile.close()
print('Xong')