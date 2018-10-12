import SVM
import argparse
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("division number", help='please set division number to divide training data', type=int)
    parser.add_argument("kernel", help='select from "no", "polynomial", "gauss"', type=str)
    parser.add_argument("file", help='input data file name', type=str)
    parser.add_argument("--d", type=float, help='for polynomial kernel', default=2)
    parser.add_argument("--sigma", type=float,  help='for gauss kernel', default=3)
    parser.add_argument("--c", type=float,  help='slack variable for soft margin', default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    kernel = SVM.set_kernel(args)