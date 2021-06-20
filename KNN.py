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

# datalatih = pd.DataFrame(data_latih)
# datauji = pd.DataFrame(data_uji)

# datalatih.to_excel('./output_latih.xlsx', index=False)
# datauji.to_excel('./output_uji.xlsx', index=False)

# print(len(data_latih))
# print(len(data_uji))
# print(f"jarak row 1 dan 2 : {euclidean_distance(data[0],data[1])}")
# print(f"jarak row 1 dan 6 : {euclidean_distance(data[0],data[5])}")
# print(f"jarak row 1 dan 10 : {euclidean_distance(data[0],data[9])}")
# print(f"jarak row 1 dan 15 : {euclidean_distance(data[0],data[14])}")
# print(f"jarak row 1 dan 20 : {euclidean_distance(data[0],data[19])}")
# print(f"jarak row 1 dan 25 : {euclidean_distance(data[0],data[24])}")
# print(f"jarak row 1 dan 30 : {euclidean_distance(data[0],data[29])}")
# print(f"jarak row 1 dan 100 : {euclidean_distance(data[0],data[99])}")
# print(f"jarak row 1 dan 110 : {euclidean_distance(data[0],data[109])}")
# print(f"jarak row 1 dan 120 : {euclidean_distance(data[0],data[119])}")
# print(f"jarak row 1 dan 130 : {euclidean_distance(data[0],data[129])}")
# print(f"jarak row 1 dan 140 : {euclidean_distance(data[0],data[139])}")
# print(f"jarak row 1 dan 150 : {euclidean_distance(data[0],data[149])}")
# print(f"jarak row 1 dan 200 : {euclidean_distance(data[0],data[199])}")
# print(f"jarak row 1 dan 300 : {euclidean_distance(data[0],data[299])}")
# print(f"jarak row 1 dan 400 : {euclidean_distance(data[0],data[399])}")
# print(f"jarak row 1 dan 500 : {euclidean_distance(data[0],data[499])}")


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


num_neighbors = 3

label = predict_classification(data_latih, data_uji[200], num_neighbors)
print('Predicted: %s' % (label))
label = predict_classification(data_latih, data_uji[250], num_neighbors)
print('Predicted: %s' % (label))
label = predict_classification(data_latih, data_uji[50], num_neighbors)
print('Predicted: %s' % (label))
label = predict_classification(data_latih, data_uji[150], num_neighbors)
print('Predicted: %s' % (label))