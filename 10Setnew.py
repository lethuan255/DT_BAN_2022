#Thư viện sử dụng
import pandas as pd
import sys
import copy
#-----------------

# Lấy 10% và sau đó xóa đi những dữ liệu đã lấy ở dữ liệu cũ.
def ten_percent(assay, end, files):
    if files == 9:
        return labels_dict[assay]
    data_percent = labels_dict[assay][:end]
    if labels_dict[assay].empty:
            return data_percent
    labels_dict[assay].drop(range(0, end), inplace=True)
    labels_dict[assay] = labels_dict[assay].reset_index(drop=True)
    return data_percent

if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1], header=None)
    # Sắp xếp dữ liệu theo nhãn tăng dần
    data = data.sort_values([0, 0], ascending=[True, True])
    data = data.reset_index(drop=True)
    #Tìm họ lớn nhất và nhỏ nhất, dựa vào giá trị nhãn
    min_assay, max_assay = data[0].min(), data[0].max()
    # Tạo một Dictionary trong đó nhãn là key và giá trị của key đó là Dataframe
    labels = list(sorted(set(data.loc[:, 0])))
    labels_dict = dict([(i, pd.DataFrame()) for i in labels])
    # Các cặp số vị trí đầu mỗi họ và số lượng file của mỗi họ.
    numbs_assay = []
    for i in labels:
        idx = data.loc[data[0] == i].index[0]
        labels_number = int(len(data.loc[data[0] == i]) * 0.1)
        if labels_number == 0:
            labels_number = 1
        numbs_assay.append((idx, labels_number))

    # Add dữ liệu của mỗi nhãn vào dictionary theo đúng nhãn.
    for idx in range(len(labels)):
        start = numbs_assay[idx][0]
        if labels[idx] == max_assay:
            labels_dict[labels[idx]] = labels_dict[labels[idx]].append(data.loc[start:]).reset_index(drop=True)
        else:
            end = numbs_assay[idx + 1][0] - 1
            labels_dict[labels[idx]] = labels_dict[labels[idx]].append(data.loc[start:end]).reset_index(drop=True)

    # chia làm 10 phần bằng nhau, mỗi phần là 10% của mỗi nhãn
    files = 0
    train = []
    while files < 10:
        file_csv = pd.DataFrame()
        file_csv = file_csv.append([ten_percent(labels[idx], numbs_assay[idx][1],
                                                files)
                                   for idx in range(len(labels))])
        file_csv = file_csv.reset_index(drop=True)
        train.append(file_csv)
        file_csv.to_csv("file-{}.csv".format(files), index=False)
        files += 1
        print("Created File-{}!".format(files))
    #Tao file train
    print(len(train))
    for i in range(0, 10):
        file_csv = pd.DataFrame()
        train_temp = copy.copy(train)
        if i < 9:
            train_temp.pop(i)
            train_temp.pop(i)
        else:
            train_temp.pop(i)
            train_temp.pop(0)
        file_csv = file_csv.append(train_temp).reset_index(drop=True)
        file_csv.to_csv("train-{}.csv".format(i), index=False)
        print("Create Train-{}!".format(i))

# Các file sẽ được lưu ở thư mục thực thi code này

