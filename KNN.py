# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
import pandas as pd
from math import sqrt


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


# Calculate the Minkowski distance between two vectors
def minkowski_distance(row1, row2):
    distance = 0.0
    for i in range(2, len(row1)):
        distance += abs(row1[i] - row2[i])
    return distance


#proses normalisasi data
data_minmax = dataset_minmax(data)
normalize_dataset(data, data_minmax)

#proses normalisasi submit
data_minmax = dataset_minmax(submit)
normalize_dataset(submit, data_minmax)

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
def get_neighbors(train, test_row, num_neighbors, algorithm):
    distances = list()
    if algorithm == 'e':
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
    elif algorithm == 'm':
        for train_row in train:
            dist = minkowski_distance(test_row, train_row)
            distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors, algorithm):
    neighbors = get_neighbors(train, test_row, num_neighbors, algorithm)
    output_values = [row[1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def classification_testing(train, test, num_neighbors, algorithm):
    result = []
    accuracy = 1
    for i in range(1, len(test)):
        label = predict_classification(train, test[i], num_neighbors,
                                       algorithm)
        label_actual = test[i][1]
        result.append([label, label_actual])
        if label == label_actual:
            accuracy += 1
    return accuracy / (len(test) - 1)


def optimal_k_value(train, test, k_start, k_step, k_max, algorithm):
    max_acc = 0
    optimal_k = 1
    for i in range(k_start, k_max + 1, k_step):
        accuracy = classification_testing(train, test, i, algorithm)
        print(f"Test-{i} result : {accuracy} , for K = {i}")
        if accuracy > max_acc:
            max_acc = accuracy
            optimal_k = i
    return optimal_k, max_acc


def classification_output_latih(train, data, num_neighbors, algorithm):
    result = []
    accuracy = 1
    for i in range(1, len(data)):
        label = predict_classification(train, data[i], num_neighbors,
                                       algorithm)
        label_actual = data[i][1]
        result.append([i, label, label_actual])
        if label == label_actual:
            accuracy += 1
    result[0].append(accuracy / (len(data) - 1))
    result = pd.DataFrame(result)
    result.columns = [
        'idData', 'Klasifikasi', 'Klasifikasi Sebenarnya', 'Akurasi'
    ]
    result.to_excel('./OutputLatih.xlsx', index=False)


def classification_output_submit(train, data, num_neighbors, algorithm):
    result = []
    for i in range(1, len(data)):
        label = predict_classification(train, data[i], num_neighbors,
                                       algorithm)
        result.append([i, label])

    result = pd.DataFrame(result)
    result.columns = ['idData', 'Klasifikasi']
    result.to_excel('./OutputSubmit.xlsx', index=False)


#main program

k_start = 1
k_interval = 1
k_max = 20
accuracy = 0
k_value = 0
algorithm = ''

print("-------------------------------------------------")
k_value_e, accuracy_e = optimal_k_value(data_latih, data_uji, k_start,
                                        k_interval, k_max, 'e')

print("-------------------------------------------------")
print("Best k value (euclidean) : ", k_value_e)
print("Best accuracy (euclidean) : ", accuracy_e)

print("-------------------------------------------------")
k_value_m, accuracy_m = optimal_k_value(data_latih, data_uji, k_start,
                                        k_interval, k_max, 'm')

print("-------------------------------------------------")
print("Best k value (minkowski) : ", k_value_m)
print("Best accuracy (minkowski) : ", accuracy_m)
print("-------------------------------------------------")
if accuracy_m > accuracy_e:
    print(f"Highest accuracy : {accuracy_m} ,k = {k_value_m}, using minkowski")
    accuracy = accuracy_m
    k_value = k_value_m
    algorithm = 'm'
elif accuracy_e > accuracy_m:
    print(f"Highest accuracy : {accuracy_e} ,k = {k_value_e}, using euclidean")
    accuracy = accuracy_e
    k_value = k_value_e
    algorithm = 'e'
else:
    print(f"Highest accuracy : {accuracy_e} ,k = {k_value_m}")
    accuracy = accuracy_m
    k_value = k_value_m
    algorithm = 'm'
print("-------------------------------------------------")

classification_output_latih(data_latih, data_uji, k_value, algorithm)
classification_output_submit(data_latih, submit, k_value, algorithm)
