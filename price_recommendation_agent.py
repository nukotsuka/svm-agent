from SVM import set_kernel
from SVR import classifier
from pylab import *
import pandas
import random

# global variables
file_name = "listing_data/Hong_Kong-listings.csv"
kernel_name = "gauss"
columns = ["price", "accommodates", "beds", "bathrooms", "bedrooms"]
data_num = 300 * 3
epsilon = 50
C_list = [100, 250, 500]
sigma_range = np.arange(1, 11, 1)
price_min_limit = 100
price_max_limit = 1500


def setup_data():
    """
    Read data from listing csv file
    And divide it into train, validation, test data
    X means feature params and Y means price
    :return: train, validation, test data of X and Y
    """

    listing_data = pandas.read_csv(file_name, usecols=columns).dropna()
    listing_data = listing_data[listing_data.price.str.startswith('$')]
    listing_data.price = listing_data.price.str.replace('$', '').str.replace(',', '').astype(float)
    listing_data = listing_data[price_min_limit < listing_data.price]
    listing_data = listing_data[listing_data.price < price_max_limit]
    listing_data = listing_data.head(data_num)
    Y = listing_data.price.values.reshape((-1, 1))
    X = listing_data.drop(columns=['price']).values.astype(float)

    standardized_X = (X - X.mean()) / X.std()
    divided_X = np.split(standardized_X, 3)
    divided_Y = np.split(Y, 3)

    print("----------price description----------")
    print(listing_data.price.describe())

    return divided_X[0], divided_X[1], divided_X[2], divided_Y[0], divided_Y[1], divided_Y[2]


def train(X_train, X_validation, Y_train, Y_validation):
    """
    Create SVR by train data and params
    And validate which SVR is optimal by grid search
    :return: optimal SVR
    """

    print("----------training start----------")
    ideal_total_income = np.sum(Y_validation)
    print("ideal_total_income =", ideal_total_income)

    min_total_error = sys.float_info.max
    optimal_params = {"C": C_list[0], "sigma": sigma_range[0]}
    optimal_SVR = (lambda x: x)

    for C in C_list:
        for sigma in sigma_range:
            kernel = set_kernel(kernel_name, d=2, sigma=sigma)
            SVR, _, _, _ = classifier(X_train, Y_train, C, epsilon, kernel)

            total_income = 0
            total_error = 0
            for i in range(len(X_validation)):
                predict_price = SVR(X_validation[i])
                correct_price = Y_validation[i]

                total_error += (predict_price - correct_price) ** 2

                if (predict_price == correct_price and random.randint(0, 1) == 0) or predict_price < correct_price:
                    total_income += predict_price

            if total_error < min_total_error:
                min_total_error = total_error
                optimal_params = {"C": C, "sigma": sigma}
                optimal_SVR = SVR
            print("C = {0:4}, sigma = {1:4}, total_income = {2}, error = {3}".format(C, sigma, total_income, total_error / len(X_validation)))

    print("----------training finish----------")
    print("optimal_params =", optimal_params)
    return optimal_SVR


def evaluate(SVR, X_test, Y_test):
    print("----------evaluation start----------")
    ideal_total_income = np.sum(Y_test)
    total_income = 0
    success_count = 0
    for i in range(len(X_test)):
        predict_price = SVR(X_test[i])[0] * 0.90
        correct_price = Y_test[i][0]
        if (predict_price == correct_price and random.randint(0, 1) == 0) or predict_price < correct_price:
            total_income += predict_price
            success_count += 1
        # print("predict_price = {0}, correct_price = {1}, diff = {2}".format(predict_price, correct_price, correct_price - predict_price))

    print("----------evaluation finish----------")
    print("success_count = {0}, total_income = {1}, ideal_total_income = {2}, ratio = {3}".format(success_count, total_income, ideal_total_income, total_income / ideal_total_income))


def simple_strategy(Y_test):
    print("----------simple strategy----------")
    price_mean = Y_test.mean()
    print("price_mean =", price_mean)
    total_income = 0
    for i in range(len(Y_test)):
        correct_price = Y_test[i][0]
        if (price_mean == correct_price and random.randint(0, 1) == 0) or price_mean < correct_price:
            total_income += price_mean

    print("total_income =", total_income)

def main():
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = setup_data()
    SVR = train(X_train, X_validation, Y_train, Y_validation)
    evaluate(SVR, X_test, Y_test)
    simple_strategy(Y_test)


if __name__ == '__main__':
    main()