from SVM import set_kernel
import cvxopt
import argparse
from pylab import *
import os



def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help='select from "no", "polynomial", "gauss"', type=str)
    parser.add_argument("file", help='input data file name', type=str)
    parser.add_argument("--d", type=float, help='for polynomial kernel', default=2)
    parser.add_argument("--sigma", type=float,  help='for gauss kernel', default=np.sqrt(5))
    parser.add_argument("--c", type=float,  help='slack variable for soft margin', default=1000)
    parser.add_argument("--epsilon", type=float,  help='for epsilon insensitive function', default=0.1)
    args = parser.parse_args()
    return args


def set_data(args):
    data = np.loadtxt(args.file, delimiter=",")
    D = len(data[0][:-1])  # dimension of data point
    X = data[:, :D]
    Y = data[:, D:]
    return data, D, X, Y


def lagrange(X, Y, C, epsilon, kernel):
    N = len(X)
    tmp = np.zeros((N * 2, N * 2))
    for i in range(N):
        for j in range(N):
            tmp[i, j] = kernel(X[i], X[j])
    P = cvxopt.matrix(tmp)
    tmp = np.zeros((N, N))
    for i in range(N):
        tmp[i, i] = -Y[i]
    q = cvxopt.matrix(np.hstack((-np.squeeze(Y, 1), epsilon * np.ones(N))))
    G = cvxopt.matrix(np.vstack((
        np.hstack((np.diag([0.5] * N), np.diag([0.5] * N))),
        np.hstack((np.diag([-0.5] * N), np.diag([0.5] * N))),
        np.hstack((np.diag([-0.5] * N), np.diag([-0.5] * N))),
        np.hstack((np.diag([0.5] * N), np.diag([-0.5] * N)))
    )))
    h = cvxopt.matrix(np.hstack((C * np.ones(N * 2), np.zeros(N * 2))))
    A = cvxopt.matrix(np.hstack((np.ones(N), np.zeros(N))), (1, N * 2), 'd')
    b = cvxopt.matrix(0.0)

    cvxopt.solvers.options['abstol'] = 1e-5
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['show_progress'] = False
    result = cvxopt.solvers.qp(P, q, G=G, h=h, A=A, b=b, kktsolver='ldl')
    x = np.array(result['x']).reshape(N * 2)
    A = (x[:N] + x[N:]) / 2
    A_star = (x[N:] - x[:N]) / 2
    # print(np.hstack((A, A_star)))
    return A, A_star


def classifier(X, Y, C, epsilon, kernel):
    A, A_star = lagrange(X, Y, C, epsilon, kernel)

    S = []
    S_star = []

    for i in range(len(A)):
        if epsilon < A[i] < C - epsilon:
            S.append(i)

    for i in range(len(A_star)):
        if epsilon < A_star[i] < C - epsilon:
            S_star.append(i)

    w = 0
    for k in np.hstack((S, S_star)):
        k_ = int(k)
        w += (A[k_] - A_star[k_]) * X[k_]

    sum = 0
    for n in S:
        tmp = 0
        for k in range(len(A)):
            tmp += (A[k] - A_star[k]) * kernel(X[n], X[k])
        sum += -Y[n] + epsilon + tmp

    for n in S_star:
        tmp = 0
        for k in range(len(A_star)):
            tmp += (A[k] - A_star[k]) * kernel(X[n], X[k])
        sum += -Y[n] - epsilon + tmp

    if len(np.hstack((S, S_star))) == 0:
        theta = 0
    else:
        theta = sum / len(np.hstack((S, S_star)))

    def f(x):
        sum = 0.0
        for k in np.hstack((S, S_star)):
            k_ = int(k)
            sum += (A[k_] - A_star[k_]) * kernel(X[k_], x)
        return sum - theta

    return f, S, w, theta


def show_result(X, Y, f, epsilon):
    for i in range(len(X)):
        predict = f(X[i])
        if np.abs(predict - Y[i]) > epsilon:
            print('support vector:', i)

    total_square_error = 0
    for i in range(len(X)):
        predict = f(X[i])
        square_error = ((predict - Y[i]) ** 2)[0]
        total_square_error += square_error
        print('data: {0}, answer: {1}, predict: {2}, square error: {3}'.format(X[i], Y[i][0], predict[0], square_error))

    print('average square error:', total_square_error / len(X))


def plot(args, data, f, S):
    for n in S:
        plt.plot(data[n,0], data[n,1], 'bo', markersize=10)

    for d in data:
        plt.plot(d[0], d[1], 'kx')


    X1, X2 = meshgrid(linspace(0, 1.2, 300), linspace(0, 1.2, 300))
    w, h = X1.shape
    X1.resize(X1.size)
    X2.resize(X2.size)
    Z = array([f(array([x1, x2])) for (x1, x2) in zip(X1, X2)])
    X1.resize((w, h))
    X2.resize((w, h))
    Z.resize((w, h))
    contours = contour(X1, X2, Z, linewidths=1, origin='lower')
    clabel(contours, inline=1, fontsize=10)

    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)

    file = str(args.file).replace('sample_', '').replace('.txt', '')
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists('./result'):
        os.mkdir('./result')
    plt.savefig('./result/' + args.kernel + '_' + file + date + '.png')
    print(args.kernel + '_' + file + date + '.png is saved in result folder.')
    print('please open:')
    print('open result/' + args.kernel + '_' + file + date + '.png')



def main():
    args = set_args()
    kernel = set_kernel(args.kernel, args.d, args.sigma)
    data, D, X, Y = set_data(args)
    f, S, w, theta = classifier(X, Y, args.c, args.epsilon, kernel)

    print('w =', w)
    print('θ =', theta)

    show_result(X, Y, f, args.epsilon)

    if D == 2:
        plot(args, data, f, S)


if __name__ == '__main__':
    main()