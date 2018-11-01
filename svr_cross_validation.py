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
    parser.add_argument("--epsilon", type=float, help='for epsilon insensitive function', default=0.1)
    parser.add_argument("--c_list", type=list,  help='slack variable list', default=[1, 10, 100, 1000])
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
    epsilon = args.epsilon

    X, Y, D = divide_data(file_name, N)

    C_list = args.c_list
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

    kernel_param_points, C_points = np.meshgrid(kernel_param_range, C_list)

    loss_matrix = np.zeros((len(C_list), len(kernel_param_range)))
    min_loss = 100000000000
    min_loss_params = {}
    for i in range(len(kernel_param_range)):
        for j in range(len(C_list)):
            kernel_param = kernel_param_points[j][i]
            C = C_points[j][i]

            kernel = set_kernel(kernel_name, d=kernel_param, sigma=kernel_param)

            total_squares_loss = 0
            for n in range(N):
                X_ = X.copy()
                Y_ = Y.copy()
                X_test = X_.pop(n)
                Y_test = Y_.pop(n)
                X_train = np.vstack(X_)
                Y_train = np.vstack(Y_)

                f, S = classifier(X_train, Y_train, C, epsilon, kernel)

                squares_loss = 0
                for i in range(len(X_test)):
                    squares_loss += (f(X_test[i]) - Y_test[i]) ** 2
                total_squares_loss +=  squares_loss / len(X_test)

            average_squares_loss = total_squares_loss / N
            print(kernel_param_name, '=', kernel_param, 'C =', C, 'average_squares_loss =', average_squares_loss)
            loss_matrix[j][i] = average_squares_loss
            if average_squares_loss < min_loss:
                min_loss = average_squares_loss
                min_loss_params = {kernel_param_name: kernel_param, 'C': C}

    print('min loss =', min_loss)
    print('min loss params =', min_loss_params)


def main():
    args = set_args()
    cross_validation(args)



if __name__ == '__main__':
    main()