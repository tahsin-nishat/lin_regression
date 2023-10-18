# Author: Tahsin Nishat (PhD Student, CAEM, University of Arizona)
# Contact: nishat@arizona.edu
# Date: 10.18.23

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Set the following lines to True in order to make the script execute
RUN_FIT = True

# Data directory
DATA_ROOT = 'data'
PATH_TO_DATA = os.path.join(DATA_ROOT, 'rand_data.txt')

FIGURES_ROOT = 'figures'
PATH_TO_FIG = os.path.join(FIGURES_ROOT, 'fitted model.png')


def rand_data_generation(N):
    """
    This function generates some random data for a polynomial function
    :param N: number of data points
    :return data: generated input and output data (with NX2 size)
    """

    # Generate evenly spaced input features from 0 to 10 with N number of data points
    x=np.array([np.linspace(1, 10, N)]).T
    # Generate true output data from input for a polynomial model f(x)=35-2x+3x^2+0.01x^3
    y=35-2*x+3*x**2+0.01*x**3
    y=y+np.random.rand(N,1)*20 # Add gaussian noise to the output data
    data=np.c_[x,y]
    return data

def linear_regression(x_train,y_train,maxorder=5):
    """
    This function implements the linear regression model from training data
    for a polynomial fit i.e., y=w0+w1*x+w2*x^2+......+wn*x^n
    In matrix form Y=X.W
    where: Y=output matrix (NxD), X=Design matrix (NXD), W=Weight matrix (DX1)
    For a polynomial fit by MSE, the normal equation is: W=(XT.X)^-1*XT.Y

    :param  : x_train = input features (1D array)
            : y_train = output (1D array)
            : maxorder = maximum model order considered to find the best fit

    :return : best_poly = obtained best polynomial order
            : best_poly_w = best parameters for the fitted model
    """
    train_loss = np.zeros((x_train.shape[0], maxorder + 1))  # log the training loss
    # Design matrix representation of input features
    for i in range(maxorder+1):
        X=np.zeros((x_train.shape[0],i+1))

        for k in range(i+1):
            X[:,k]=np.power(x_train,k)

        # Calculate w vector (as an numpy.array)
        w=(np.linalg.inv(X.T@X)@X.T)@y_train
        # model predictions on training data
        predict_y=X@w

        # Calculate MSE (Mean Squared Error) loss
        train_loss[:, i] = np.mean(np.power(predict_y- y_train, 2))


    # Ensure taking log of the mean (not mean of the log!)
    mean_train_loss = np.mean(train_loss, axis=0)
    # The loss values can get quite large, so take the log for display purposes
    log_mean_train_loss = np.log(mean_train_loss)
    # Find the minimum loss
    min_mean_loss = np.min(log_mean_train_loss)

    # Obtaining the parameters for minimum training loss
    best_poly = [i for i, j in enumerate(log_mean_train_loss) if j == min_mean_loss][0]

    # Calculating the parameters for best polynomial model order
    X = np.zeros((x_train.shape[0], best_poly + 1))
    for k in range(best_poly + 1):
        X[:, k] = np.power(x_train, k)

    best_model_w = (np.linalg.inv(X.T @ X) @ X.T) @ y_train

    return best_poly,best_model_w

def fitpoly(data_path, figure_path):

    # Only run the data_generation if no data available
    N=1000
    data=rand_data_generation(N)
    np.savetxt(data_path, data,delimiter=',')

    # Comment out the above lines and uncomment the following to load
    # the available dataset from the data directory
    # data=np.loadtxt(data_path) # (Nx2) array
    # where N= number of data points, col1=input, col2=output

    N=len(data)
    # Shuffle the training data. This is very important!
    data=shuffle(data)
    # Split the data into train and test
    x_train=data[:int(7/10*N),0] # taking 7/10 data for training
    y_train = data[:int(7/10*N), 1]  # taking 7/10 data for training
    x_test = data[int(7/10*N):, 0]  # taking 3/10 data for test
    y_test = data[int(7/10*N):, 1]  # taking 3/10 data for test


    # Fit the data set and find the best polynomial order
    order,w = linear_regression(x_train,y_train,maxorder=10)

    Xtest = np.zeros((x_test.shape[0], order + 1))
    for k in range(order + 1):
        Xtest[:, k] = np.power(x_test, k)

    # model predictions on test data
    y_predict = Xtest@w

    # To Visualize the predictions and True output
    plt.figure()
    plt.scatter(x_test,y_predict,s=5)
    plt.scatter(x_test, y_test,s=5,alpha=0.5)
    plt.title('Predicted Output vs. True Output (Best Model Order = '+str(order)+')')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(["Predictions from the fitted model","True output"])
    plt.savefig(figure_path,format='png')


if __name__ == '__main__':
    if RUN_FIT:
        fitpoly(PATH_TO_DATA,PATH_TO_FIG)
        plt.show()
