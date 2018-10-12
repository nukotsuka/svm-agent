import cvxopt
from scipy.linalg import norm
import matplotlib
matplotlib.use('Agg')
from pylab import *
import argparse
import datetime
import os
from sklearn.preprocessing import StandardScaler


def no_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(d):
    return (lambda x, y: (1 + np.dot(x, y)) ** d)


def gaussian_kernel(sigma):
    return (lambda x, y: np.exp(-norm(x - y) ** 2 / (2 * (sigma ** 2))))

def lagrange(X, Y, C, kernel):
    N = len(X)
    tmp = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            tmp[i, j] = Y[i] * Y[j] * kernel(X[i], X[j])
    P = cvxopt.matrix(tmp)
    q = cvxopt.matrix(-np.ones(N))
    G = cvxopt.matrix(np.vstack((np.diag([-1.0]*N), np.identity(N))))
    h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
    A = cvxopt.matrix(Y, (1, N), 'd')
    b = cvxopt.matrix(0.0)

    cvxopt.solvers.options['abstol'] = 1e-5
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['show_progress'] = False
    result = cvxopt.solvers.qp(P, q, G=G, h=h, A=A, b=b, kktsolver='ldl')
    A = np.array(result['x']).reshape(N)
    return A


def classifier(X, Y, D, C, kernel):
    A = lagrange(X, Y, C, kernel)

    S = []  # support vectors
    M = []  # margins
    for i in range(len(A)):
        if 0 < A[i]:
            S.append(i)
        if 0 < A[i] < C:
            M.append(i)

    w = np.zeros(D)
    for i in S:
        w += A[i] * Y[i] * X[i]

    sum = 0
    for i in M:
        tmp = 0
        for j in S:
            tmp += A[j] * Y[j] * kernel(X[i], X[j])
        sum += (Y[i] - tmp)
    theta = sum / len(M)

    print('w =', w)
    print('θ =', theta)

    def f(x):
        sum = 0.0
        for n in S:
            sum += A[n] * Y[n] * kernel(x, X[n])
        return sum + theta

    return f


def plot(args, data, f):
    for d in data:
        if d[-1] == 1:
            plt.plot(d[0], d[1], 'rx')
        else:
            plt.plot(d[0], d[1], 'bx')

    X1, X2 = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100))
    w, h = X1.shape
    X1.resize(X1.size)
    X2.resize(X2.size)
    Z = array([f(array([x1, x2])) for (x1, x2) in zip(X1, X2)])
    X1.resize((w, h))
    X2.resize((w, h))
    Z.resize((w, h))
    contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    file = str(args.file).replace('sample_', '').replace('.txt', '')
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('./result'):
        os.mkdir('./result')
    plt.savefig('./result/' + args.kernel + '_' + file + date + '.png')
    print(args.kernel + '_' + file + date + '.png is saved in result folder.')
    print('please open:')
    print('open result/' + args.kernel + '_' + file + date + '.png')

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help='select from "no", "polynomial", "gauss"', type=str)
    parser.add_argument("file", help='input data file name', type=str)
    parser.add_argument("--d", type=float, help='for polynomial kernel', default=2)
    parser.add_argument("--sigma", type=float,  help='for gauss kernel', default=3)
    parser.add_argument("--c", type=float,  help='slack variable for soft margin', default=0.5)
    args = parser.parse_args()
    return args


def set_kernel(args):
    if args.kernel == 'polynomial':
        return polynomial_kernel(d=args.d)
    elif args.kernel == 'gauss':
        return gaussian_kernel(sigma=args.sigma)
    elif args.kernel == 'no':
        return no_kernel
    else:
        raise Exception(args.kernel + ' is invalid input. select from "no", "polynomial", "gauss".')


def set_data(args):
    data = np.loadtxt(args.file, delimiter=",")
    D = len(data[0][:-1])  # dimension of data point
    X = data[:, :D]
    Y = data[:, D:]
    sc = StandardScaler()
    sc.fit(X)
    X_ = sc.transform(X)
    return np.hstack((X_, Y)), D, X_, Y


def main():
    args = set_args()
    kernel = set_kernel(args)
    data, D, X, Y = set_data(args)
    f = classifier(X, Y, D, args.c, kernel)

    if D == 2:
        plot(args, data, f)


if __name__ == '__main__':
    main()
