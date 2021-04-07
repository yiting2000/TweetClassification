# TweetClassification
An old school project
The aim is to predict the location of twitter users using the tweets they generated. 

We learn to process text data using nltk and the n-gram model.
The baseline model used is the naive bayes classifier implemnented directly from sklearn APIs.
We then further improving the model by 
1) the use of random forest and other ensemble methods 
2) the use of embeddings generated from pretrained models (due to the lack of data, the embedding method does not work as well as we expected).



Usage: 

main.py:
Outputs the final test results.

models: the models used to improve the results 

featureSelection.py:
Uses Regex, countVectorizer and the Chi2 Test to identify the k best ngrams in terms of their score in the Chi2 test.


formatData.py:
Takes a given set of feature names, the set of training tweets and the set of testing/validation tweets and constructs Xtrain, Ytrain, Xtest and Ytest.

model.py:
Constructs the model, trains it on the training data and tests it on the validation data.


Tweet Files:
Samples of the tweet txt files have also been included.
