# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
import pandas as pd
from math import sqrt


def getdata(filename, sheet):
    xlsx = pd.read_excel(filename, sheet)
    return xlsx


filename = 'DataSetTB3_SHARE.xlsx'
data = getdata(filename, 0)  #data uji dan data latih
submit = getdata(filename, 1)  #data yang dicari
data = data.values.tolist()


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(2, len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(2, len(row)):
            print(f"I ke-{i} , panjang  {len(row)}")
            if minmax[i - 2][1] == 0:
                row[i] = 0
            else:
                row[i] = (row[i] - minmax[i - 2][0]) / (minmax[i - 2][1] -
                                                        minmax[i - 2][0])


data_minmax = dataset_minmax(data)
normalize_dataset(data, data_minmax)
print(data)

# # Convert string column to float
# def str_column_to_float(dataset, column):
#     for row in dataset:
#         row[column] = float(row[column].strip())

# # Convert string column to integer
# def str_column_to_int(dataset, column):
#     class_values = [row[column] for row in dataset]
#     unique = set(class_values)
#     lookup = dict()
#     for i, value in enumerate(unique):
#         lookup[value] = i
#         print('[%s] => %d' % (value, i))
#     for row in dataset:
#         row[column] = lookup[row[column]]
#     return lookup

# # Calculate the Euclidean distance between two vectors
# def euclidean_distance(row1, row2):
#     distance = 0.0
#     for i in range(len(row1) - 1):
#         distance += (row1[i] - row2[i])**2
#     return sqrt(distance)

# # Locate the most similar neighbors
# def get_neighbors(train, test_row, num_neighbors):
#     distances = list()
#     for train_row in train:
#         dist = euclidean_distance(test_row, train_row)
#         distances.append((train_row, dist))
#     distances.sort(key=lambda tup: tup[1])
#     neighbors = list()
#     for i in range(num_neighbors):
#         neighbors.append(distances[i][0])
#     return neighbors

# # Make a prediction with neighbors
# def predict_classification(train, test_row, num_neighbors):
#     neighbors = get_neighbors(train, test_row, num_neighbors)
#     output_values = [row[-1] for row in neighbors]
#     prediction = max(set(output_values), key=output_values.count)
#     return prediction

# Make a prediction with KNN on Iris Dataset

# for i in range(len(dataset[0]) - 1):
#     str_column_to_float(dataset, i)
# # convert class column to integers
# str_column_to_int(dataset, len(dataset[0]) - 1)
# # define model parameter
# num_neighbors = 5
# # define a new record
# row = [5.7, 2.9, 4.2, 1.3]
# # predict the label
# label = predict_classification(dataset, row, num_neighbors)
# print('Data=%s, Predicted: %s' % (row, label))