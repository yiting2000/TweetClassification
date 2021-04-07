""" modified from 
http://scikit-learn.org/dev/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from scipy import stats

import codecs
import regex
import numpy as np
import pickle

from nltk.stem import *
stemmer = SnowballStemmer("english")

TRAIN_FILE_NAME = "train-tweets.txt"
FEATURE_NAMES_FILE="feature_names.pkl"

N_FEATURES = 150000
LENGTH_NGRAM = 1

def stem_tweet(tweet):
    """
    Passed the tweet, splits it into individual then stems
    
    **This was disabled as in a number of trials undertaken stemming did not
    improve performance.
    """
    tweet = tweet.split(' ')
    tweet = [stemmer.stem(word) for word in tweet]
    tweet = ' '.join(tweet)
    return tweet

def pre_process_tweet(tweet):
    """Performs regex operations to strip undesirable elements of the
    tweet."""
    #Converts tweet to lower case
    tweet = tweet.lower()

    #**Disabled, was initially recommended to remove hyperlinks,
    #but it was found that these features were quite 'good.'

    #remove hyperlinks
    #url_regex = regex.compile("http.*\s?")
    #tweet = regex.sub(url_regex,"",tweet)

    #removes the location from the end of the tweet
    tweet = regex.sub("\w*$","",tweet)

    #removes the tweet ID, through positive lookbehind
    tweet = regex.sub("(?<=(^\d*))\s\d*","",tweet)

    #removes all numbers, excluding the UserID, which is now first
    #through negative lookbehind
    tweet = regex.sub("(?<!(^\d*))\d*","",tweet)

    #removes non-alpha-numeric characters
    tweet = regex.sub("[^a-zA-Z0-9\s]","",tweet)

    #replaces any multi-white-space with a single space
    tweet = regex.sub("\s+"," ",tweet)

    #removes leading or trailing whitespace
    tweet = regex.sub("(^\s)|(\s$)","",tweet)

    #Stemming Disabled
    #tweet = stem_tweet(tweet)

    return tweet

def processData(file_name,X=None,Y=None):
    """
    If X and Y are None, proceeds to open the indicated file.
    Reads the tweets contained within extracting information
    on a per user basis.
    It is an easier task to classifier the location of users, by
    combining all of the tweets of each user. This simplification
    of the problem works as users typically stay in the same location.
    """
    if (X==None and Y==None):        
        train_file = codecs.open(file_name,'r','utf-8')

        tweets = [line for line in train_file]

        if file_name==TRAIN_FILE_NAME:
            X=[]
            #TODO: Could be optimized
            for tweet in tweets:
                instance = pre_process_tweet(tweet)
                message = regex.sub("\d","",instance)
                X.append(message)
                
            print(X[0])
            #retrieves location tag
            Y = [regex.search("\w*$",tweet).group(0) for tweet in tweets]
            Y = np.asarray(Y)
            print(Y[0])

        else:
            per_user_data = {}
            for tweet in tweets:
                #retrieves location tag
                Y = regex.search("\w*$",tweet).group(0)

                instance = pre_process_tweet(tweet)
                UserID = regex.search("^\d*",instance).group(0)
                message = instance[len(UserID)+1:]

                if UserID in per_user_data:
                    #add to existing user
                    per_user_data[UserID][0] += " "+message                
                    per_user_data[UserID][1].append(Y)
                else:
                    #create new user
                    per_user_data[UserID] = ['',[Y]]
                    per_user_data[UserID][0] = message

            #for each tweet, determines the user and sets X to be their combined tweets
            X = [per_user_data[regex.search("^\d*",tweet).group(0)][0] for tweet in tweets]
            print(X[0])

            #Sets Y to be the location tag
            Y = [regex.search("\w*$",tweet).group(0) for tweet in tweets]        
            Y = np.asarray(Y)
            print(Y[0])

    return X,Y
def featureSelection(n_features,length_ngram,X=None,Y=None):
    """
    Uses Chi2 test to identify 'good' features in X.
    **The collection of p_values was for report writing.
    """
    X,Y = processData(TRAIN_FILE_NAME,X,Y)

    vectorizer = CountVectorizer(decode_error='ignore',ngram_range=(1,length_ngram))
    X = vectorizer.fit_transform(X)

    feature_names = vectorizer.get_feature_names()

    selected = SelectKBest(chi2,k=min(n_features,len(feature_names)))
    selected.fit_transform(X, Y)

    feature_names = [feature_names[i] for i in selected.get_support(indices=True)]

    #Try and except remnant of using another non-probabilistic feature selector
    try:
        p_values = np.sort(selected.pvalues_)[0:n_features]
    except:
        p_values = [1]*len(feature_names)

    print(feature_names[:min(100,n_features)])

    return p_values, feature_names

if __name__ == "__main__":
    p_values, feature_names = featureSelection(N_FEATURES,LENGTH_NGRAM)

    store = open(FEATURE_NAMES_FILE, 'wb')
    pickle.dump(feature_names, store)
    store.close()
