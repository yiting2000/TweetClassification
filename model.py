import pickle
import numpy as np
from time import time
from scipy.stats import mode
import matplotlib.pyplot as plt
from collections import Counter 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

X_TRAIN_FILE = "X_TRAIN.pkl"
Y_TRAIN_FILE = "Y_TRAIN.pkl"

X_TEST_FILE = "X_TEST.pkl"
Y_TEST_FILE = "Y_TEST.pkl"

#Found using Grid Search, on training data

#Optimal Parameters for Multinomial
OPT_ALPHA=0.204


def model(Xtrain,Ytrain,Xtest,Ytest):
    """
    Constructs and trains a multinomial Naive Bayes Learner on the training
    data, then produces performance metrics on the test data. 
    """
    
    model = MultinomialNB(alpha=OPT_ALPHA)
    model.fit(Xtrain,Ytrain)

    pred = model.predict(Xtest)

    i=0
    for a,b in zip(pred,Ytest):
        if i>20:
            break
        i+=1
        print(a,b)

    R0_Baseline = mode(Ytest)[1]/len(Ytest)
    print("Accuracy: ",metrics.accuracy_score(Ytest,pred))
    print(metrics.accuracy_score(Ytest,pred)*(len(Ytest)))
    print("Baseline: ",R0_Baseline," Guessing ",mode(Ytest)[0])
    print(metrics.classification_report(Ytest,pred))

    """
    #hyper-parameter optimization for MultinomialNB
    parameters = {'alpha':np.linspace(-10,10,50)}
    meta_opt = GridSearchCV(model, parameters)
    meta_opt.fit(Xtrain, Ytrain)
    print(meta_opt.best_params_)
    """     


    return metrics.accuracy_score(Ytest,pred),Ytest,pred

def plot_confusion_matrix(cm,locations, title='Confusion matrix', cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(locations))
    plt.xticks(tick_marks, locations, rotation=45)
    plt.yticks(tick_marks, locations)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__=="__main__":
    store = open(X_TRAIN_FILE, 'rb')
    Xtrain = pickle.load(store)    
    store.close()

    store = open(Y_TRAIN_FILE, 'rb')
    Ytrain = pickle.load(store)    
    store.close()

    store = open(X_TEST_FILE, 'rb')
    Xtest = pickle.load(store)    
    store.close()

    store = open(Y_TEST_FILE, 'rb')
    Ytest = pickle.load(store)    
    store.close()

    accuracy,yTest,yPred = model(Xtrain,Ytrain,Xtest,Ytest)
    cm = metrics.confusion_matrix(yTest,yPred)
    print(cm)

    locations = ["B","H","SD","Se","W"]

    plot_confusion_matrix(cm,locations)

    print(Counter(Ytrain))
