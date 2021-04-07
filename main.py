import matplotlib.pyplot as plt
import codecs
import regex
import numpy as np

import featureSelection
import model
import formatData

TRAIN_FILE_NAME = "train-tweets.txt"
TEST_FILE_NAME = "dev-tweets.txt"
#TEST_FILE_NAME = "test-tweets.txt"

RESULT_FILE_NAME = 'test-results.txt'

MAX_NGRAM=1

if __name__=="__main__":
    #n_features = [50,100,250,500,1000,2500,5000,10000,25000,50000,100000,150000,
    #200000]
    #n_features = [1.5*10**5,2*10**5,5*10**5,7.5*10**5,1*10**6,1*10**6,1.5*10**6,2*10**6,2.5*10**6,3*10**6,3.5*10**6]
    n_features = [150000]

    #a list to store the accuracies of the trials
    accuracy = [ [0]*len(n_features) for i in range(MAX_NGRAM)]
    p_vals = [0]*MAX_NGRAM

    Xtrain,Ytrain = featureSelection.processData(TRAIN_FILE_NAME)
    Xtest,Ytest = featureSelection.processData(TEST_FILE_NAME)
    test_file = codecs.open(TEST_FILE_NAME,'r','utf-8')
    tweets = [line for line in test_file]
    tweet_ids = [regex.search("(?<=(^\d*\s))\d*",tweet).group(0) for tweet in tweets]

    for n in range(1,MAX_NGRAM+1):
        for i in range(0,len(n_features)):
            #only about 2300000 length_ngram=1 features
            if n==1 and n_features[i]>300000:
                accuracy[n-1][i] = accuracy[n-1][i-1]
                continue
            print("Ngram:",n,"n_features:", n_features[i])
            #only need last p_vals
            p_vals[n-1],feature_names = featureSelection.featureSelection(n_features[i],n,Xtrain,Ytrain)
            X1,Y1,X2,Y2 = formatData.formatData(feature_names,Xtrain,Ytrain,Xtest,Ytest)
            accuracy[n-1][i],y1,y2 = model.model(X1,Y1,X2,Y2)
    
    print(p_vals)
    print(accuracy)
    print(n_features)

    plt.figure(1)
    #p_value subplot
    for n in range(0,MAX_NGRAM):
        plt.subplot(211)
        plt.plot(p_vals[n])
    plt.xlabel('Number of Features')
    plt.legend(['Marginal P-Value of Feature-Ngram=1','Marginal P-Value of Feature-Ngram=2',
        'Marginal P-Value of Feature-Ngram=3',],loc='upper left')

    #Accuracy subplot
    for n in range(0,MAX_NGRAM):
        plt.subplot(212)
        plt.plot(n_features,accuracy[n])
    plt.xlabel('Number of Features')
    plt.legend(['Accuracy of Classifier-Ngram=1','Accuracy of Classifier-Ngram=2',
        'Accuracy of Classifier-Ngram=3'],loc='lower right')
    plt.show()

    #output results, format of output dictated by assessment
    paired = [str(a)+","+str(b) for a,b in zip(tweet_ids,y2)]

    out_file = open(RESULT_FILE_NAME,'w')

    #Header
    out_file.write("Id,Category\n")
    
    for val in paired:
        out_file.write(val)
        out_file.write('\n')
