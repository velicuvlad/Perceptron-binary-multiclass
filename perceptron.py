
# Name: Vlad-Alexandru Velicu
# SID: 201348604

from csv import reader
import numpy as np
import copy

def load_data(file):
    dataset = list()
    with open(file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def convert_strings_to_floats(dataset):
    for row in dataset:
        for column in range(4):
            row[column] = float(row[column])
    return dataset

def convert_classes_to_binary(dataset):
    for i in range(len(dataset)):
        if(i < len(dataset) / 2):
            dataset[i][-1] = 1
        else: 
            dataset[i][-1] = -1
    return dataset

def convert_classes_to_binary_multiclass(class_name, dataset):
    clone = copy.deepcopy(dataset)
    for i in range(len(clone)):
        if(clone[i][-1] == class_name):
            clone[i][-1] = 1
        else: 
            clone[i][-1] = -1
    return clone

def split_data_based_on_class(dataset):
    datasets = {'class-1': [], 'class-2': [], 'class-3': []}
    for i in range(len(dataset)):
        label = dataset[i][-1]
        datasets[label].append(dataset[i])
    return datasets

def predict_binary(row, weights, bias):
	activation = bias
	for i in range(len(row)-1):
		activation += weights[i] * row[i]
	return 1 if activation >= 0 else -1

def predict_multiclass(row, weights, bias):
    activation = bias
    for i in range(len(row)-1):
	    activation += weights[i] * row[i]
    return activation

def train_binary(dataset, epochs):
    weights = [0.0 for i in range(4)]
    bias = 0
    total_correct = 0
    for epoch in range(epochs):
        correct = 0
        for row in dataset:
            label = row[-1]
            activation = predict_binary(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] += (label - activation) * row[i]
                bias += label - activation
            else:
                correct += 1
                total_correct += 1
    acc = total_correct / (len(dataset) * epochs) * 100
    print("Train accuracy: %d%%" % acc)
    return [bias, weights]

def train_multiclass(dataset_1, dataset_2, dataset_3, epochs, regularisation=0):
    weights = [[0.0 for i in range(4)], [0.0 for i in range(4)], [0.0 for i in range(4)]]
    bias = [0, 0, 0]
    total_correct = 0
    for epoch in range(epochs):
        for row in dataset_1:
            label = row[-1]
            activation = predict_binary(row, weights[0], bias[0])
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[0][i] = (1 - 2 * regularisation) * weights[0][i] + (label - activation) * row[i]
                bias[0] += label - activation
        for row in dataset_2:
            label = row[-1]
            activation = predict_binary(row, weights[1], bias[1])
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[1][i] = (1 - 2 * regularisation) * weights[1][i] + (label - activation) * row[i]
                bias[1] += label - activation
        for row in dataset_3:
            label = row[-1]
            activation = predict_binary(row, weights[2], bias[2])
            if(label * activation <= 0):
                for i in range(len(row)-1):
                   weights[2][i] = (1 - 2 * regularisation) * weights[2][i] + (label - activation) * row[i]
                bias[2] += label - activation
    return [bias, weights]

def test_binary(dataset, trained_weights, bias):
    correct = 0
    for row in dataset:
        label = row[-1]
        activation = predict_binary(row, trained_weights, bias)
        if(label * activation > 0):
            correct += 1
    acc = correct / (len(dataset)) * 100
    return acc
    
def test_multiclass(dataset, trained_weights, bias):
    activation=[0, 0, 0]
    correct = 0
    for row in dataset['class-1']:
        activation[0] = predict_multiclass(row, trained_weights[0], bias[0])
        activation[1] = predict_multiclass(row, trained_weights[1], bias[1])
        activation[2] = predict_multiclass(row, trained_weights[2], bias[2])
        if(max(activation) == activation[0]):
            correct += 1
    for row in dataset['class-2']:
        label = row[-1]
        activation[0] = predict_multiclass(row, trained_weights[0], bias[0])
        activation[1] = predict_multiclass(row, trained_weights[1], bias[1])
        activation[2] = predict_multiclass(row, trained_weights[2], bias[2])
        if (max(activation) == activation[1]):
            correct += 1
    for row in dataset['class-3']:
        label = row[-1]
        activation[0] = predict_multiclass(row, trained_weights[0], bias[0])
        activation[1] = predict_multiclass(row, trained_weights[1], bias[1])
        activation[2] = predict_multiclass(row, trained_weights[2], bias[2])
        if (max(activation) == activation[2]):
            correct += 1

    acc = correct / (len(dataset['class-1'])+len(dataset['class-2'])+len(dataset['class-3'])) * 100
    return acc

def run_binary_perceptron(split_train_dataset, split_test_dataset, training_epochs):
    print('=========================================================')
    print('Training perceptron to discriminate between class 1 and class 2')
    train_data_split_1 = convert_classes_to_binary(split_train_dataset['class-1'] + split_train_dataset['class-2']) 
    bias_1, weights_1 = train_binary(train_data_split_1, training_epochs)

    test_data_split_1 = convert_classes_to_binary(split_test_dataset['class-1'] + split_test_dataset['class-2'])
    print("Test accuracy: %d%%" % test_binary(test_data_split_1, weights_1, bias_1))
    print('=========================================================')

    print('Training perceptron to discriminate between class 2 and class 3')
    train_data_split_2 = convert_classes_to_binary(split_train_dataset['class-2'] + split_train_dataset['class-3']) 
    bias_2, weights_2 = train_binary(train_data_split_2, training_epochs)

    test_data_split_2 = convert_classes_to_binary(split_test_dataset['class-2'] + split_test_dataset['class-3'])
    print("Test accuracy: %d%%" % test_binary(test_data_split_2, weights_2, bias_2))
    print('=========================================================')

    print('Training perceptron to discriminate between class 1 and class 3')
    train_data_split_3 = convert_classes_to_binary(split_train_dataset['class-1'] + split_train_dataset['class-3']) 
    bias_3, weights_3 = train_binary(train_data_split_3, training_epochs)

    test_data_split_3 = convert_classes_to_binary(split_test_dataset['class-1'] + split_test_dataset['class-3'])
    print("Test accuracy: %d%%" % test_binary(test_data_split_3, weights_3, bias_3))
    print('=========================================================')

def run_multiclass_perceptron(train_dataset, split_train_dataset, split_test_dataset, training_epochs, regularisation=0):
    print('=========================================================')
    print('Training perceptron to discriminate between class 1 and the rest')
    print('Training perceptron to discriminate between class 2 and the rest')
    print('Training perceptron to discriminate between class 3 and the rest')

    convert_1 = convert_classes_to_binary_multiclass('class-1', train_dataset)

    convert_2 = convert_classes_to_binary_multiclass('class-2', train_dataset)

    convert_3 = convert_classes_to_binary_multiclass('class-3', train_dataset)

    bias, weights = train_multiclass(convert_1,  convert_2, convert_3, training_epochs, regularisation)
    print('=========================================================')

    print("Accuracy for the whole Train dataset: %d%%" % test_multiclass(split_train_dataset, weights, bias))
    print("Accuracy for the whole Test dataset: %d%%" %   test_multiclass(split_test_dataset, weights, bias))
   
    print('=========================================================')


print('\nQuestion 3')
train_dataset_q3 = convert_strings_to_floats(load_data('train.data'))
np.random.shuffle(train_dataset_q3)
split_train_dataset_q3 = split_data_based_on_class(train_dataset_q3)

test_dataset_q3 = convert_strings_to_floats(load_data('test.data'))
np.random.shuffle(test_dataset_q3)
split_test_dataset_q3 = split_data_based_on_class(test_dataset_q3)

run_binary_perceptron(split_train_dataset_q3, split_test_dataset_q3, 20)

print('\nQuestion 4')
train_dataset_q4 = convert_strings_to_floats(load_data('train.data'))
np.random.shuffle(train_dataset_q4)
split_train_dataset_q4 = split_data_based_on_class(train_dataset_q4)

test_dataset_q4 = convert_strings_to_floats(load_data('test.data'))
np.random.shuffle(test_dataset_q4)
split_test_dataset_q4 = split_data_based_on_class(test_dataset_q4)

run_multiclass_perceptron(train_dataset_q4, split_train_dataset_q4, split_test_dataset_q4, 20)

print('\nQuestion 5')
print('Regularisation 0.01')
run_multiclass_perceptron(train_dataset_q4, split_train_dataset_q4, split_test_dataset_q4, 20, 0.01)
print('\nRegularisation 0.1')
run_multiclass_perceptron(train_dataset_q4, split_train_dataset_q4, split_test_dataset_q4, 20, 0.1)
print('\nRegularisation 1.0')
run_multiclass_perceptron(train_dataset_q4, split_train_dataset_q4, split_test_dataset_q4, 20, 1.0)
print('\nRegularisation 10.0')
run_multiclass_perceptron(train_dataset_q4, split_train_dataset_q4, split_test_dataset_q4, 20, 10.0)
print('\nRegularisation 100.0')
run_multiclass_perceptron(train_dataset_q4, split_train_dataset_q4, split_test_dataset_q4, 20, 100.0)



