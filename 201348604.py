from csv import reader
import numpy as np

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
    for i in range(len(dataset)):
        if(dataset[i][-1] == class_name):
            dataset[i][-1] = 1
        else: 
            dataset[i][-1] = -1
    return dataset

def split_data_based_on_class(dataset):
    datasets = {'class-1': [], 'class-2': [], 'class-3': []}
    for i in range(len(dataset)):
        label = dataset[i][-1]
        datasets[label].append(dataset[i])
    return datasets

def predict(row, weights, bias):
	activation = bias
	for i in range(len(row)-1):
		activation += weights[i] * row[i]
	return 1 if activation >= 0 else -1

def train_perceptron(dataset, epochs):
    weights = [0.0 for i in range(4)]
    bias = 0
    total_correct = 0
    for epoch in range(epochs):
        correct = 0
        for row in dataset:
            label = row[-1]
            activation = predict(row, weights, bias)
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

def train_multiclass_perceptron(dataset_1, dataset_2, dataset_3, epochs):
    weights = [0.0 for i in range(4)]
    bias = 0
    total_correct = 0
    for epoch in range(epochs):
        for row in dataset_1:
            label = row[-1]
            activation = predict(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] += (label - activation) * row[i]
                bias += label - activation
        for row in dataset_2:
            label = row[-1]
            activation = predict(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] += (label - activation) * row[i]
                bias += label - activation
        for row in dataset_3:
            label = row[-1]
            activation = predict(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] += (label - activation) * row[i]
                bias += label - activation
    return [bias, weights]

def train_multiclass_regularisation(dataset_1, dataset_2, dataset_3, epochs, l):
    weights = [0.0 for i in range(4)]
    bias = 0
    total_correct = 0
    for epoch in range(epochs):
        for row in dataset_1:
            label = row[-1]
            activation = predict(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] = (1 - 2 * l)*weights[i] + (label - activation) * row[i]
                bias += label - activation
        for row in dataset_2:
            label = row[-1]
            activation = predict(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] = (1 - 2 * l)*weights[i] + (label - activation) * row[i]
                bias += label - activation
        for row in dataset_3:
            label = row[-1]
            activation = predict(row, weights, bias)
            if(label * activation <= 0):
                for i in range(len(row)-1):
                    weights[i] = (1 - 2 * l)*weights[i] + (label - activation) * row[i]
                bias += label - activation
    print('Bias: ', bias)
    print('Weights: ', weights)
    return [bias, weights]

def test_perceptron(dataset, trained_weights, bias):
    correct = 0
    for row in dataset:
        label = row[-1]
        activation = predict(row, trained_weights, bias)
        if(label * activation > 0):
            correct += 1
    acc = correct / (len(dataset)) * 100
    return acc

def run_binary_train_and_test(split_train_dataset, split_test_dataset, training_epochs):
    print('=========================================================')
    print('Training perceptron to discriminate between class 1 and class 2')
    train_data_split_1 = convert_classes_to_binary(split_train_dataset['class-1'] + split_train_dataset['class-2']) 
    np.random.shuffle(train_data_split_1)
    weights_1 = train_perceptron(train_data_split_1, training_epochs)

    test_data_split_1 = convert_classes_to_binary(split_test_dataset['class-1'] + split_test_dataset['class-2'])
    np.random.shuffle(test_data_split_1)
    print("Test accuracy: %d%%" % test_perceptron(test_data_split_1, weights_1[1], weights_1[0]))
    print('=========================================================')

    print('Training perceptron to discriminate between class 2 and class 3')
    train_data_split_2 = convert_classes_to_binary(split_train_dataset['class-2'] + split_train_dataset['class-3']) 
    np.random.shuffle(train_data_split_2)
    weights_2 = train_perceptron(train_data_split_2, training_epochs)

    test_data_split_2 = convert_classes_to_binary(split_test_dataset['class-2'] + split_test_dataset['class-3'])
    np.random.shuffle(test_data_split_2)
    print("Test accuracy: %d%%" % test_perceptron(test_data_split_2, weights_2[1], weights_2[0]))
    print('=========================================================')

    print('Training perceptron to discriminate between class 1 and class 3')
    train_data_split_3 = convert_classes_to_binary(split_train_dataset['class-1'] + split_train_dataset['class-3']) 
    np.random.shuffle(train_data_split_3)
    weights_3 = train_perceptron(train_data_split_3, training_epochs)

    test_data_split_3 = convert_classes_to_binary(split_test_dataset['class-1'] + split_test_dataset['class-3'])
    np.random.shuffle(test_data_split_3)
    print("Test accuracy: %d%%" % test_perceptron(test_data_split_3, weights_3[1], weights_3[0]))
    print('=========================================================')

"""def run_multiclass_clasification(train_dataset, test_dataset, training_epochs):
    print('\nTraining perceptron to discriminate between class 1 and the rest')
    train_data_split_1 = convert_classes_to_binary_multiclass('class-1', train_dataset) 
    np.random.shuffle(train_data_split_1)
    weights_1 = train_perceptron(train_data_split_1, training_epochs)

    test_data_split_1 = convert_classes_to_binary_multiclass('class-1', test_dataset)
    np.random.shuffle(test_data_split_1)
    test_perceptron(test_data_split_1, weights_1[1], weights_1[0])
    print('=========================================================')

    print('Training perceptron to discriminate between class 2 and the rest')
    train_data_split_2 = convert_classes_to_binary_multiclass('class-2',train_dataset) 
    np.random.shuffle(train_data_split_2)
    weights_2 = train_perceptron(train_data_split_2, training_epochs)

    test_data_split_2 = convert_classes_to_binary_multiclass('class-2', test_dataset)
    np.random.shuffle(test_data_split_2)
    test_perceptron(test_data_split_2, weights_2[1], weights_2[0])
    print('=========================================================')

    print('Training perceptron to discriminate between class 3 and the rest')
    train_data_split_3 = convert_classes_to_binary_multiclass('class-3', train_dataset) 
    np.random.shuffle(train_data_split_3)
    weights_3 = train_perceptron(train_data_split_3, training_epochs)

    test_data_split_3 = convert_classes_to_binary_multiclass('class-3', test_dataset)
    np.random.shuffle(test_data_split_3)
    test_perceptron(test_data_split_3, weights_3[1], weights_3[0])
    print('=========================================================')
"""

def run_multiclass_clasification(train_dataset, test_dataset, training_epochs):
    print('=========================================================')
    print('Training perceptron to discriminate between class 1 and the rest')
    print('Training perceptron to discriminate between class 2 and the rest')
    print('Training perceptron to discriminate between class 3 and the rest')
    train_data_split_1 = convert_classes_to_binary_multiclass('class-1', train_dataset)
    np.random.shuffle(train_data_split_1)

    train_data_split_2 = convert_classes_to_binary_multiclass('class-2',train_dataset)
    np.random.shuffle(train_data_split_2)

    train_data_split_3 = convert_classes_to_binary_multiclass('class-3', train_dataset)
    np.random.shuffle(train_data_split_3)
    
    weights = train_multiclass_perceptron(train_data_split_1,  train_data_split_2, train_data_split_3, training_epochs)
    print('=========================================================')

    np.random.shuffle(train_dataset)
    np.random.shuffle(test_dataset)
    print("Accuracy for the whole Train dataset: %d%%" % test_perceptron(train_dataset, weights[1], weights[0]))
    print("Accuracy for the whole Test dataset: %d%%" %  test_perceptron(test_dataset, weights[1], weights[0]))
   
    print('=========================================================')

def run_multiclass_classification_regularisation(train_dataset, test_dataset, training_epochs, l):
    print('=========================================================')
    print('Training perceptron to discriminate between class 1 and the rest using regularisation')
    print('Training perceptron to discriminate between class 2 and the rest using regularisation')
    print('Training perceptron to discriminate between class 3 and the rest using regularisation')
    train_data_split_1 = convert_classes_to_binary_multiclass('class-1', train_dataset) 
    np.random.shuffle(train_data_split_1)

    train_data_split_2 = convert_classes_to_binary_multiclass('class-2',train_dataset) 
    np.random.shuffle(train_data_split_2)

    train_data_split_3 = convert_classes_to_binary_multiclass('class-3', train_dataset) 
    np.random.shuffle(train_data_split_3)

    weights = train_multiclass_regularisation(train_data_split_1,  train_data_split_2, train_data_split_3, training_epochs, l)
    print('=========================================================')

    np.random.shuffle(train_dataset)
    np.random.shuffle(test_dataset)
    print("Accuracy for the whole Train dataset: %d%%" % test_perceptron(train_dataset, weights[1], weights[0]))
    print("Accuracy for the whole Test dataset: %d%%" %  test_perceptron(test_dataset, weights[1], weights[0]))
   
    print('=========================================================')


train_data = convert_strings_to_floats(load_data('train.data'))
split_train_data = split_data_based_on_class(train_data)

test_data = convert_strings_to_floats(load_data('test.data'))
split_test_data = split_data_based_on_class(test_data)

print('Question 3')
run_binary_train_and_test(split_train_data, split_test_data, 20)

print('\nQuestion 4')
run_multiclass_clasification(train_data, test_data, 20)

print('\nQuestion 5')
print('Regularisation 0.01')
run_multiclass_classification_regularisation(train_data, test_data, 20, 0.01)
print('\nRegularisation 0.1')
run_multiclass_classification_regularisation(train_data, test_data, 20, 0.1)
print('\nRegularisation 1.0')
run_multiclass_classification_regularisation(train_data, test_data, 20, 1.0)
print('\nRegularisation 10.0')
run_multiclass_classification_regularisation(train_data, test_data, 20, 10.0)
print('\nRegularisation 100.0')
run_multiclass_classification_regularisation(train_data, test_data, 20, 100.0)



