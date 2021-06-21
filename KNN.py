# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
import pandas as pd
from math import sqrt
import random


def getdata(filename, sheet):
    xlsx = pd.read_excel(filename, sheet)
    return xlsx


filename = 'DataSetTB3_SHARE.xlsx'
data = getdata(filename, 0)  #data uji dan data latih
submit = getdata(filename, 1)  #data yang dicari
data = data.values.tolist()  #data dirubah dari bentuk dataframe menjadi list
submit = submit.values.tolist()


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(
            2, len(dataset[0])
    ):  #rangenya dari 2 sampai length karena 2 kolom pertama tidak digunakan (id dan klasifikasi)
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(2, len(row)):
            if minmax[i - 2][1] == 0:
                row[i] = 0
            else:
                row[i] = (row[i] - minmax[i - 2][0]) / (minmax[i - 2][1] -
                                                        minmax[i - 2][0])
                #indeks minmax dikurangi 2 karena minmax[0] merupakan informasi minmax dari
                #dataset[2], dst.


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(2, len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


#proses normalisasi data
data_minmax = dataset_minmax(data)
normalize_dataset(data, data_minmax)

#membagi data menjadi data latih (70%) dan data uji (30%)
data_uji = []
data_latih = []
temp = 0
x = 0
for i in range(len(data)):
    if (data[i][1] == temp) and (x < 30):
        data_uji.append(data[i])
        x += 1
    elif x >= 30:
        x = 0
        temp += 1
        data_latih.append(data[i])
    else:
        data_latih.append(data[i])


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def classification_testing(train, test, test_iter, num_neighbors):
    result = []
    accuracy = 1
    for i in range(test_iter):
        n = random.randrange(1, 300)
        label = predict_classification(train, test[n], num_neighbors)
        label_actual = test[n][1]
        result.append([label, label_actual])
        if label == label_actual:
            accuracy += 1
    return accuracy / test_iter


def optimal_k_value(train, test, test_iter, k_step, k_max):
    max_acc = 0
    optimal_k = 1
    for i in range(1, k_max, k_step):
        accuracy = classification_testing(train, test, test_iter, i)
        if accuracy > max_acc:
            max_acc = accuracy
            optimal_k = i
    return optimal_k, max_acc


k_value, accuracy = optimal_k_value(data_latih, data_uji, 100, 1, 50)
print("Best k value : ", k_value)
print("Best accuracy : ", accuracy)

# num_neighbors = 3
# label = predict_classification(data_latih, data_uji[200], num_neighbors)
# print('Predicted: %s' % (label))
# label = predict_classification(data_latih, data_uji[250], num_neighbors)
# print('Predicted: %s' % (label))
# label = predict_classification(data_latih, data_uji[50], num_neighbors)
# print('Predicted: %s' % (label))
# label = predict_classification(data_latih, data_uji[150], num_neighbors)
# print('Predicted: %s' % (label))