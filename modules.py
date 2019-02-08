from sklearn.naive_bayes import binarize as bn

def predict(data):
    result = bn(data,0,copy=True)
    return (result)