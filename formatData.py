import pickle
import codecs
import regex
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

import featureSelection as fs

TRAIN_FILE_NAME = "train-tweets.txt"
FEATURE_NAMES_FILE="feature_names.pkl"

TEST_FILE_NAME = "dev-tweets.txt"

X_TRAIN_FILE = "X_TRAIN.pkl"
Y_TRAIN_FILE = "Y_TRAIN.pkl"

X_TEST_FILE = "X_TEST.pkl"
Y_TEST_FILE = "Y_TEST.pkl"


def vectorizeData(file_name,feature_names,X=None,Y=None):
    X,Y = fs.processData(file_name,X,Y)

    vectorizer=CountVectorizer(decode_error='ignore',vocabulary=feature_names)
    X = vectorizer.fit_transform(X)
    return X,Y

def formatData(feature_names,Xtrain=None,Ytrain=None,Xtest=None,Ytest=None):
    X1,Y1 = vectorizeData(TRAIN_FILE_NAME,feature_names,Xtrain,Ytrain)

    X2,Y2 = vectorizeData(TEST_FILE_NAME,feature_names,Xtest,Ytest)

    return X1,Y1,X2,Y2


if __name__=="__main__":
    store = open(FEATURE_NAMES_FILE, 'rb')
    feature_names = pickle.load(store)    
    store.close()

    X1,Y1,X2,Y2 = formatData(feature_names)

    store = open(X_TRAIN_FILE, 'wb')
    pickle.dump(X1, store)
    store.close()

    store = open(Y_TRAIN_FILE, 'wb')
    pickle.dump(Y1, store)
    store.close()

    store = open(X_TEST_FILE, 'wb')
    pickle.dump(X2, store)
    store.close()

    store = open(Y_TEST_FILE, 'wb')
    pickle.dump(Y2, store)
    store.close()
