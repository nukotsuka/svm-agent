import SVM
import numpy as np
import argparse
import matplotlib


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help='division number to divide training data', type=int)
    parser.add_argument("kernel", help='select from "no", "polynomial", "gauss"', type=str)
    parser.add_argument("file", help='input data file name', type=str)
    parser.add_argument("--d_from", type=float, help='for polynomial kernel', default=1)
    parser.add_argument("--d_to", type=float, help='for polynomial kernel', default=4)
    parser.add_argument("--sigma_from", type=float,  help='for gauss kernel', default=2)
    parser.add_argument("--sigma_to", type=float,  help='for gauss kernel', default=5)
    parser.add_argument("--c_from", type=float,  help='slack variable for soft margin', default=0.2)
    parser.add_argument("--c_to", type=float,  help='slack variable for soft margin', default=1.0)
    args = parser.parse_args()
    return args


def divide_data(file_name, N):
    data = np.loadtxt(file_name, delimiter=",")
    D = len(data[0][:-1])  # dimension of data point
    np.random.shuffle(data)
    X = data[:, :D]
    Y = data[:, D:]
    standardized_X = (X - X.mean()) / X.std()
    divided_X = np.split(standardized_X, N)
    divided_Y = np.split(Y, N)

    return divided_X, divided_Y, D

def cross_validation():
    args = set_args()
    file_name = args.file
    N = args.n  # division number
    C = args.c_from
    kernel = SVM.set_kernel(args)
    X, Y, D = divide_data(file_name, N)

    total_accuracy = 0
    for i in range(N):
        X_ = X.copy()
        Y_ = Y.copy()
        X_test = X_.pop(i)
        Y_test = Y_.pop(i)
        X_train = np.vstack(X_[i] for i in range(len(X_)))
        Y_train = np.vstack(Y_[i] for i in range(len(Y_)))

        f = SVM.classifier(X_train, Y_train, D, C, kernel)

        predict_result = []
        for x in X_test:
            if f(x) > 0:
                predict_result.append(1)
            else:
                predict_result.append(-1)

        accuracy = (len(Y_test) - np.sum((Y_test.reshape(-1) + predict_result) == 0)) / len(Y_test)
        print(i, ': accuracy =', accuracy)
        total_accuracy += accuracy

    print('accuracy: ', total_accuracy / N)


if __name__ == '__main__':
    cross_validation()