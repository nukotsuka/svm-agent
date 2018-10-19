from SVM import set_kernel, classifier
import argparse
import matplotlib
matplotlib.use('Agg')
from pylab import *
import os
import datetime


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help='division number to divide training data', type=int)
    parser.add_argument("kernel", help='select from "polynomial", "gauss"', type=str, choices=['polynomial', 'gauss'])
    parser.add_argument("file", help='input data file name', type=str)
    parser.add_argument("--d_from", type=float, help='start point of d range for polynomial kernel', default=1)
    parser.add_argument("--d_to", type=float, help='end point of d range for polynomial kernel', default=10)
    parser.add_argument("--d_interval", type=float, help='interval of d range for polynomial kernel', default=1)
    parser.add_argument("--sigma_from", type=float,  help='start point of sigma range for gauss kernel', default=0.5)
    parser.add_argument("--sigma_to", type=float,  help='end point of sigma range for gauss kernel', default=5)
    parser.add_argument("--sigma_interval", type=float,  help='interval of sigma range for gauss kernel', default=0.5)
    parser.add_argument("--c_from", type=float,  help='start point of slack variable range for soft margin', default=0.1)
    parser.add_argument("--c_to", type=float,  help='end point of slack variable range for soft margin', default=5)
    parser.add_argument("--c_interval", type=float,  help='interval of slack variable range for soft margin', default=0.1)
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
    kernel_name = args.kernel
    file_name = args.file
    N = args.n  # division number
    X, Y, D = divide_data(file_name, N)

    C_range = np.arange(args.c_from, args.c_to + args.c_interval, args.c_interval)
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

    kernel_param_points, C_points = np.meshgrid(kernel_param_range, C_range)

    accuracy_matrix = np.zeros((len(C_range), len(kernel_param_range)))
    max_accuracy = 0
    max_accuracy_params = {}
    for i in range(len(kernel_param_range)):
        for j in range(len(C_range)):
            kernel_param = kernel_param_points[j][i]
            C = C_points[j][i]

            kernel = set_kernel(kernel_name, d=kernel_param, sigma=kernel_param)

            total_accuracy = 0
            for n in range(N):
                X_ = X.copy()
                Y_ = Y.copy()
                X_test = X_.pop(n)
                Y_test = Y_.pop(n)
                X_train = np.vstack(X_)
                Y_train = np.vstack(Y_)

                f = classifier(X_train, Y_train, D, C, kernel)

                predict_result = []
                for x in X_test:
                    if f(x) > 0:
                        predict_result.append(1)
                    else:
                        predict_result.append(-1)

                accuracy = (len(Y_test) - np.sum((Y_test.reshape(-1) + predict_result) == 0)) / len(Y_test)
                total_accuracy += accuracy

            average_accuracy = total_accuracy / N
            print(kernel_param_name, '=', kernel_param, 'C =', C, 'accuracy =', average_accuracy)
            accuracy_matrix[j][i] = average_accuracy
            if average_accuracy > max_accuracy:
                max_accuracy = average_accuracy
                max_accuracy_params = {kernel_param_name: kernel_param, 'C': C}

    print('max accuracy =', max_accuracy)
    print('max accuracy params =', max_accuracy_params)
    plot_contour(args, kernel_param_name, kernel_param_points, C_points, accuracy_matrix)


def plot_contour(args, kernel_param_name, kernel_param_points, C_points, accuracy_matrix):
    plt.axes()
    color_list = ['purple', 'navy', 'blue', 'skyblue', 'darkcyan', 'green', 'olive', 'gold', 'orange', 'red']
    CS = plt.contour(kernel_param_points, C_points, accuracy_matrix, 10, colors=color_list, linewidths=1,
                     origin='lower')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlim(kernel_param_points.min(), kernel_param_points.max())
    plt.ylim(C_points.min(), C_points.max())
    plt.xlabel(kernel_param_name, fontsize=16)
    plt.ylabel('C', fontsize=16)
    file = str(args.file).replace('sample_', '').replace('.txt', '')
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('./result'):
        os.mkdir('./result')
    plt.savefig('./result/contour_' + args.kernel + '_' + file + date + '.png')
    print('contour_' + args.kernel + '_' + file + date + '.png is saved in result folder.')
    print('please open:')
    print('open result/contour_' + args.kernel + '_' + file + date + '.png')


if __name__ == '__main__':
    cross_validation()