from sklearn.naive_bayes import GaussianNB as gnb
import numpy as np

model = gnb()

def predict(data):
    result = gnb(data,0,copy=True)
    return (result)

def fit(x,y):
    model.fit(x,y)
    return model

def prior():
    return 0

def likelihood():
    return 0

def posterior():
    return 0


def predicted():
    model.predicted()
    return 0

