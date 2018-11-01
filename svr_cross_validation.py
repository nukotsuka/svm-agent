from SVM import set_kernel
from SVR import classifier
import argparse
from pylab import *
import os


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", help='division number to divide training data', type=int, default=10)
    parser.add_argument("kernel", help='select from "no", "polynomial", "gauss"', type=str)
    parser.add_argument("file", help='input data file name', type=str)
    parser.add_argument("--d_from", type=float, help='start point of d range for polynomial kernel', default=1)
    parser.add_argument("--d_to", type=float, help='end point of d range for polynomial kernel', default=10)
    parser.add_argument("--d_interval", type=float, help='interval of d range for polynomial kernel', default=1)
    parser.add_argument("--sigma_from", type=float, help='start point of sigma range for gauss kernel', default=0.5)
    parser.add_argument("--sigma_to", type=float, help='end point of sigma range for gauss kernel', default=5)
    parser.add_argument("--sigma_interval", type=float, help='interval of sigma range for gauss kernel', default=0.5)
    parser.add_argument("-el", "--epsilon_list",  type=float, action="append", help='for epsilon insensitive function', default=[])
    parser.add_argument("-cl", "--c_list", type=int, action="append",  help='slack variable list', default=[])
    args = parser.parse_args()
    return args


def divide_data(file_name, N):
    data = np.loadtxt(file_name, delimiter=",")
    D = len(data[0][:-1])  # dimension of data point
    np.random.shuffle(data)
    X = data[:, :D]
    Y = data[:, D:]
    divided_X = np.split(X, N)
    divided_Y = np.split(Y, N)

    return divided_X, divided_Y, D


def cross_validation(args):
    N = args.n  # division number
    kernel_name = args.kernel
    file_name = args.file

    X, Y, D = divide_data(file_name, N)

    C_list = args.c_list
    epsilon_list = args.epsilon_list
    d_range = np.arange(args.d_from, args.d_to + args.d_interval, args.d_interval)
    sigma_range = np.arange(args.sigma_from, args.sigma_to + args.sigma_interval, args.sigma_interval)

    kernel_param_range = []
    kernel_param_name = ''
    if kernel_name == 'polynomial':
        kernel_param_range = d_range
        kernel_param_name = 'd'
    elif kernel_name == 'gauss':
        kernel_param_range = sigma_range
        kernel_param_name = 'sigma'
    elif kernel_name == 'no':
        kernel_param_range = [0]
        kernel_param_name = ''

    min_error = 100000000000
    min_error_params = {}
    for kernel_param in kernel_param_range:
        for C in C_list:
            for epsilon in epsilon_list:
                kernel = set_kernel(kernel_name, d=kernel_param, sigma=kernel_param)

                total_squares_error = 0
                for n in range(N):
                    X_ = X.copy()
                    Y_ = Y.copy()
                    X_test = X_.pop(n)
                    Y_test = Y_.pop(n)
                    X_train = np.vstack(X_)
                    Y_train = np.vstack(Y_)

                    f, S, w, theta = classifier(X_train, Y_train, C, epsilon, kernel)

                    squares_error = 0
                    for i in range(len(X_test)):
                        squares_error += (f(X_test[i]) - Y_test[i]) ** 2
                    total_squares_error +=  squares_error / len(X_test)

                average_squares_error = total_squares_error / N

                if kernel_name == 'no':
                    print('C = {0:4}, epsilon = {1:4}, average squares error = {2}'
                          .format(C, epsilon, average_squares_error[0]))
                else:
                    print('{0} = {1:4}, C = {2:4}, epsilon = {3:4}, average squares error = {4}'
                          .format(kernel_param_name, kernel_param, C, epsilon, average_squares_error[0]))

                if average_squares_error < min_error:
                    min_error = average_squares_error
                    if kernel_name == 'no':
                        min_error_params = {'C': C, 'epsilon': epsilon}
                    else:
                        min_error_params = {kernel_param_name: kernel_param, 'C': C, 'epsilon': epsilon}

    print('file:', file_name)
    print('kernel:', kernel_name)
    print('min average square error =', min_error[0])
    print('min average square error params =', min_error_params)


def main():
    args = set_args()
    cross_validation(args)



if __name__ == '__main__':
    main()