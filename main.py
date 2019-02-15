from sklearn.naive_bayes import GaussianNB as gnb
import numpy as np


if __name__ == '__main__':

    # assigning predictor and target variables
    x = np.array(
        [[-3, 7], [1, 5], [1, 2], [-2, 0], [2, 3], [-4, 0], [-1, 1], [1, 1], [-2, 2], [2, 7], [-4, 1], [-2, 7]])
    Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

    # Create a Gaussian Classifier
    model = gnb()

    # Train the model using the training sets
    model.fit(x, Y)

    # data
    data = [[1, 2], [3, 4]]
    # Predict Output
    predicted = model.predict(data)
    print("data : ",data,"\npredicted : ",predicted)