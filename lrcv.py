import csv
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold

threshold = 0.5


def read_data():
    # Begin with empty arrays
    data_x = []
    data_y = []

    # Read file line by line
    line_cnt = 0
    with open('covtype.libsvm.binary.scale.txt') as f:
        for line in f:
            # Split line by spaces
            cols = line.split()

            # First number is the class
            data_y.append(int(cols[0]) - 1)

            # Iterate through rest columns
            row = [0] * 20
            for col in cols[1:]:
                # Split index and value (<index>:<value>)
                nums = col.split(':')

                # Use the value and index
                idx = int(nums[0]) - 1
                val = float(nums[1])
                if idx < 20:
                    row[idx] = val

            # Append row to data_x (array of array)
            data_x.append(row)

            line_cnt += 1
            if line_cnt > 100000:
                break

    return data_x, data_y


def k_fold(data_x):
    kf = KFold(n_splits=10)
    train_idx = []
    test_idx = []
    for train, test in kf.split(data_x):
        train_idx.append(train)
        test_idx.append(test)

    return train_idx, test_idx


def lr(train_x, train_y, test_x):
    # Train model
    logistic_model = LogisticRegression()
    logistic_model.fit(train_x, train_y)

    # Test model
    model_out = logistic_model.predict_proba(test_x)[:, 1]

    prediction = []
    for i in range(0, len(model_out)):
        if(model_out[i] <= threshold):
            prediction.append(0)
        elif(model_out[i] > threshold):
            prediction.append(1)

    return prediction


# Calculate accuracy, precision, f1, recall, errot rate
def evaluate(prediction, labels):
    data_len = len(prediction)
    assert(len(labels) == data_len)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    F1 = 0
    precision = 0
    recall = 0
    accuracy = 0
    error_rate = 0
    for i in range(0, data_len):
        if(prediction[i] == 1 and labels[i] == 1):
            TP = TP + 1
        elif(prediction[i] == 1 and labels[i] == 0):
            FP = FP + 1
        elif(prediction[i] == 0 and labels[i] == 1):
            FN = FN + 1
        elif(prediction[i] == 0 and labels[i] == 0):
            TN = TN + 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    error_rate = (FP + FN) / (TP + FP + FN + TN)
    # f1 = 2 * ((precision * recall) / (precision + recall)
    F1 = (2 * TP) / (2 * TP + FP + FN)

    print("recall: %f" % recall)
    print("precision: %f" % precision)
    print("accuracy: %f" % accuracy)
    print("error_rate: %f" % error_rate)
    print("f1: %f" % F1)
    return accuracy


def main():
    # Read
    data_x, data_y = read_data()
    train_idx, test_idx = k_fold(data_x)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    all_train_x = []
    all_train_y = []
    all_test_x = []
    all_test_y = []
    accuracy = []

    for i in range(0, len(train_idx)):
        for j in range(0, len(train_idx[i])):
            train_x.append(data_x[train_idx[i][j]])
            train_y.append(data_y[train_idx[i][j]])
        all_train_x.append(train_x)
        all_train_y.append(train_y)

    for i in range(0, len(test_idx)):
        for j in range(0, len(test_idx[i])):
            test_x.append(data_x[test_idx[i][j]])
            test_y.append(data_y[test_idx[i][j]])
        all_test_x.append(test_x)
        all_test_y.append(test_y)

    for i in range(0, len(all_train_x)):
        prediction = lr(all_train_x[i], all_train_y[i], all_test_x[i])
        print("The result of round %d" % i, "is:")
        accuracy.append(evaluate(prediction, test_y))
        print('\n')

    total = 0
    for i in range(0, len(accuracy)):
        total = total + accuracy[i]
    avg_accuracy = total / len(accuracy)
    print("The average accuracy is: %f" % avg_accuracy)

main()
