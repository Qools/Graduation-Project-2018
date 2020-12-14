import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from time import time
from collections import Counter, defaultdict
import nltk
from pprint import pprint
import copy
import seaborn as sb
from IPython.core.display import HTML
from sklearn.cross_validation import train_test_split ,StratifiedKFold
from sklearn.metrics import roc_curve, auc ,confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.preprocessing import scale, MinMaxScaler, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from collections import Counter

import os
import time
from sklearn.datasets import load_files
from sklearn.cross_validation import  ShuffleSplit
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import cross_val_score

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Reading NYT dataset
df_nyt = pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/NYT_Combined_Clean_APR_4_No_Slash.csv")

# Adding label to real news to the dataset and showing the column names
df_nyt["fakeness"] = 0
df_nyt.columns

# Reading The Guardian dataset
df_tguard = pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/Clean_TheGuardian_Combined_No_Slash.csv")

# Adding label to real news to the dataset and showing the column names
df_tguard["fakeness"] = 0
df_tguard.columns

df_fake = pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/FakeNews_Clean_All.csv")

# Label was there for fake news dataset and showing the column names
df_fake.columns

# Changing the name of the column for concating later
df_tguard = df_tguard.rename(columns={'bodyText' : 'body','webPublicationDate':'pub_date'})
df_nyt = df_nyt.rename(columns={'head_clean':'headline', '_id':'id'})
df_fake = df_fake.rename(columns={'text':'body','title':'headline','uuid':'id','published':'pub_date'})
df_fake.columns,df_nyt.columns,df_tguard.columns

# Dropping unnecesary columns
df_fake.drop([u'Unnamed: 0', u'ord_in_thread', u'language', u'crawled',  u'site_url', u'country', u'thread_title', u'spam_score', u'replies_count', u'participants_count', u'likes', u'comments', u'shares', u'type'],inplace=True,axis=1)
​
df_nyt.drop([ u'Unnamed: 0', u'word_count'],inplace=True,axis=1)
​
df_tguard.drop([u'Unnamed: 0', u'apiUrl', u'fields', u'isHosted',u'pillarId', u'pillarName', u'sectionId', u'sectionName', u'type', u'webTitle', u'webUrl'],inplace=True,axis=1)

# Overview for data
df_fake.head()

# Overview for data
df_nyt.head()

# Overview for data
df_tguard.head()

# Concat the datasents
df_all = df_fake.append(df_tguard, ignore_index=True)
df_all = df_all.append(df_nyt,ignore_index=True)

#Dropping the Nan values and info
df_all.dropna(inplace=True)
print(df_all.shape)
df_all.info()

df_all.head()

df_all.to_csv("/Users/asus/source/repos/Detecting_Fake_News-master/Complete_DataSet_Clean.csv")

# Preparing the target and predictors for modeling

X_body_text = df_all.body.values
X_headline_text = df_all.headline.values
y = df_all.fakeness.values

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)

X_body_tfidf = tfidf.fit_transform(X_body_text)
X_headline_tfidf = tfidf.fit_transform (X_headline_text)

X_headline_tfidf_train, X_headline_tfidf_test, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
X_body_tfidf_train, X_body_tfidf_test, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)

class cross_validation(object):
    '''This class provides cross validation of any data set why incrementally increasing number 
       of samples in the training and test set and performing KFold splits at every iteration. 
       During cross validation the metrics accuracy, recall, precision, and f1-score are recored. 
       The results of the cross validation are display on four learning curves. '''
    
    def __init__(self, model, X_data, Y_data, X_test=None, Y_test=None, 
                 n_splits=3, init_chunk_size = 1000000, chunk_spacings = 100000, average = "binary"):

        self.X, self.Y =  shuffle(X_data, Y_data, random_state=1234)
        
        
        self.model = model
        self.n_splits = n_splits
        self.chunk_size = init_chunk_size
        self.chunk_spacings = chunk_spacings        
        
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        self.X_holdout = []
        self.Y_holdout = []
        
        self.f1_train = []
        self.f1_test = []
        self.acc_train = []
        self.acc_test = []
        self.pre_train = []
        self.pre_test = []
        self.rec_train = []
        self.rec_test = []
        
        self.f1_mean_train = []
        self.f1_mean_test = []
        self.acc_mean_train = []
        self.acc_mean_test = []
        self.pre_mean_train = []
        self.pre_mean_test = []
        self.rec_mean_train = []
        self.rec_mean_test = []
        
        self.training_size = []
        self.averageType = average
    
    def make_chunks(self):
        '''Partitions data into chunks for incremental cross validation'''
        
        # get total number of points
        self.N_total = self.X.shape[0]
        # partition data into chunks for learning
        self.chunks = list(np.arange(self.chunk_size, self.N_total, self.chunk_spacings ))
        self.remainder = self.X.shape[0] - self.chunks[-1]
        self.chunks.append( self.chunks[-1] + self.remainder )



    def train_for_learning_curve(self):
        '''KFold cross validates model and records metric scores for learning curves. 
           Metrics scored are f1-score, precision, recall, and accuracy'''

        # partiton data into chunks 
        self.make_chunks()
        # for each iteration, allow the model to use 10 more samples in the training set 
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=1234)
        # iterate through the first n samples
        for n_points in self.chunks: 
            
        
            # split the first n samples in k folds 
            for train_index, test_index in self.skf.split(self.X[:n_points], self.Y[:n_points]):
                self.train_index, self.test_index = train_index, test_index                
                self.X_train = self.X[self.train_index]
                self.X_test = self.X[self.test_index]
                self.Y_train = self.Y[self.train_index]
                self.Y_test = self.Y[self.test_index]
                
                self.model.fit(self.X_train, self.Y_train)
                self.y_pred_train = self.model.predict(self.X_train)
                self.y_pred_test = self.model.predict(self.X_test)
                self.log_metric_scores_()   
                
            self.log_metric_score_means_()
            self.training_size.append(n_points)
        
    def validate_for_holdout_set(self, X_holdout, Y_holdout):
        
        
        self.X_test = X_holdout
        self.Y_test = Y_holdout
        
        # partiton data into chunks 
        self.make_chunks()
        
        for n_points in self.chunks:
            
            self.X_train = self.X[:n_points]
            self.Y_train = self.Y[:n_points]

            self.model.fit(self.X_train, self.Y_train)
            self.y_pred_train = self.model.predict(self.X_train)
            self.y_pred_test = self.model.predict(self.X_test)
            self.log_metric_scores_()   

            self.log_metric_score_means_()
            self.training_size.append(n_points)
            
            
    
                            
    def log_metric_score_means_(self):
        '''Recrods the mean of the four metrics recording during training'''
        self.f1_mean_train.append(np.sum(self.f1_train)/len(self.f1_train))
        self.f1_mean_test.append(np.sum(self.f1_test)/len(self.f1_test))
        
        self.acc_mean_train.append(np.sum(self.acc_train)/len(self.acc_train))
        self.acc_mean_test.append(np.sum(self.acc_test)/len(self.acc_test))
        
        self.pre_mean_train.append(np.sum(self.pre_train)/len(self.pre_train))
        self.pre_mean_test.append(np.sum(self.pre_test)/len(self.pre_test))
        
        self.rec_mean_train.append(np.sum(self.rec_train)/len(self.rec_train))
        self.rec_mean_test.append(np.sum(self.rec_test)/len(self.rec_test))
        
        self.reinitialize_metric_lists_()
            
            
    def reinitialize_metric_lists_(self):
        '''Reinitializes metrics lists for training'''
        self.f1_train = []
        self.f1_test = []
        self.acc_train = []
        self.acc_test = []
        self.pre_train = []
        self.pre_test = []
        self.rec_train = []
        self.rec_test = []

            
    def log_metric_scores_(self):
        '''Records the metric scores during each training iteration'''
        self.f1_train.append(f1_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.acc_train.append(accuracy_score( self.Y_train, self.y_pred_train) )

        self.pre_train.append(precision_score(self.Y_train, self.y_pred_train, average=self.averageType))
        self.rec_train.append(recall_score( self.Y_train, self.y_pred_train, average=self.averageType) )

        self.f1_test.append(f1_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.acc_test.append(accuracy_score(self.Y_test, self.y_pred_test))

        self.pre_test.append(precision_score(self.Y_test, self.y_pred_test, average=self.averageType))
        self.rec_test.append(recall_score(self.Y_test, self.y_pred_test,average=self.averageType))
            

    def plot_learning_curve(self):
        '''Plots f1 and accuracy learning curves for a given model and data set'''
        
        fig = plt.figure(figsize = (17,12))
        # plot f1 score learning curve
        fig.add_subplot(221)   # left
        plt.title("F1-Score vs. Number of Training Samples")
        plt.plot(self.training_size, self.f1_mean_train, label="Train")
        plt.plot(self.training_size, self.f1_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("F1-Score")
        plt.legend(loc=4);
        
        # plot accuracy learning curve
        fig.add_subplot(222)   # right 
        plt.title("Accuracy vs. Number of Training Samples")
        plt.plot(self.training_size, self.acc_mean_train, label="Train")
        plt.plot(self.training_size, self.acc_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Accuracy")
        plt.legend(loc=4);
        
        # plot precision learning curve
        fig.add_subplot(223)   # left
        plt.title("Precision Score vs. Number of Training Samples")
        plt.plot(self.training_size, self.pre_mean_train, label="Train")
        plt.plot(self.training_size, self.pre_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Precision")
        plt.ylim(min(self.pre_mean_test), max(self.pre_mean_train) + 0.05)
        plt.legend(loc=4);
        
        # plot accuracy learning curve
        fig.add_subplot(224)   # right 
        plt.title("Recall vs. Number of Training Samples")
        plt.plot(self.training_size, self.rec_mean_train, label="Train")
        plt.plot(self.training_size, self.rec_mean_test, label="Test");
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Recall")
        plt.legend(loc=4);


### Assignig predictors and target values
X_body_text = df_all.body.values
X_headline_text = df_all.headline.values
y = df_all.fakeness.values

ngram_range = [(1,1),(1,2),(1,3)]
max_df = [0.65,0.75,0.85,0.90]
min_df = [0.001,0.01]
penal = ['l1','l2']

for penalt in penal:
    for gram in ngram_range:
        for mx_df in max_df:
            for mn_df in min_df:

                print ("For the parameters of \nmax_df=",mx_df,"min_df=",mn_df,"\nngram_range=",gram,"penalty as=",penalt)
                tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=gram,max_df=mx_df,min_df=mn_df)



                X_body_tfidf = tfidf.fit_transform(X_body_text)
                X_headline_tfidf = tfidf.fit_transform(X_headline_text)

                X_headline_train_tfidf, X_headline_test_tfidf, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=9876)
                X_body_train_tfidf, X_body_test_tfidf, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=9876)

                lr = LogisticRegression(penalty=penalt,n_jobs=3)
                lr.fit(X_headline_train_tfidf, y_headline_train)
                y_pred = lr.predict(X_headline_test_tfidf)

                print ("Logistig Regression F1 and Accuracy Scores : \n")
                print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_pred, average='macro')*100 ) )
                print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_headline_test, y_pred)*100) )


ngram_range = [(1,1),(1,2),(1,3)]
max_df = [0.65,0.75,0.85,0.90]
min_df = [0.001,0.01,0.1,0.25]
penal = ['l1','l2']
f1_sc_lst = []
acc_lst = []
for penalt in penal:
    for gram in ngram_range:
        for mx_df in max_df:
            for mn_df in min_df:

                print "For the parameters of \nmax_df=",mx_df,"min_df=",mn_df,"\nngram_range=",gram,"penalty as=",penalt
                tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,
                                        ngram_range=gram,max_df=mx_df,min_df=mn_df)



                X_body_tfidf = tfidf.fit_transform(X_body_text)
                #X_headline_tfidf = tfidf.fit_transform(X_headline_text)

                X_headline_train_tfidf, X_headline_test_tfidf, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
                X_body_train_tfidf, X_body_test_tfidf, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)

                lr = LogisticRegression(penalty=penalt,n_jobs=3)
                lr.fit(X_body_train_tfidf, y_body_train)
                y_pred = lr.predict(X_body_test_tfidf)

                print "Logistig Regression F1 and Accuracy Scores : \n"
                print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_pred, average='macro')*100 ) )
                print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_pred)*100) )
                
                f1_sc_lst.append(f1_score(y_body_test, y_pred, average='macro')*100)
                acc_lst.append(accuracy_score(y_body_test, y_pred)*100)

f1_sc_lst.sort()


acc_lst.sort()


lr_headline = LogisticRegression(penalty='l1')

# train model
lr_headline.fit(X_headline_tfidf_train, y_headline_train)

# get predictions for article section
y_headline_pred = lr_headline.predict(X_headline_tfidf_test)

# print metrics
print ("Logistig Regression F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_headline_pred)*100) )

cros_val_list = cross_val_score(lr_headline, X_headline_tfidf,y,cv=7)
print (cros_val_list)
print (cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)

cv = cross_validation(lr_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

lr_body = LogisticRegression(penalty='l1')

# train model
lr_body.fit(X_body_tfidf_train, y_body_train)

# get predictions for article section
y_body_pred = lr_body.predict(X_body_tfidf_test)

# print metrics
print ("Logistig Regression F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_body_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_body_pred)*100) )

cros_val_list = cross_val_score(lr_body, X_body_tfidf,y,cv=7)
print (cros_val_list)
print (cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(lr_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

rcf_headline = RandomForestClassifier(n_estimators=100,n_jobs=3)

rcf_headline.fit(X_headline_tfidf_train, y_headline_train)
y_rc_headline_pred = rcf_headline.predict(X_headline_tfidf_test)

# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_rc_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_rc_headline_pred)*100) )

cros_val_list = cross_val_score(rcf_headline, X_headline_tfidf,y,cv=5)
print (cros_val_list)
print (cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)

cv = cross_validation(rcf_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

rcf_body = RandomForestClassifier(n_estimators=100,n_jobs=3)

rcf_body.fit(X_body_tfidf_train, y_body_train)
y_rc_body_pred = rcf_body.predict(X_body_tfidf_test)

# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_rc_body_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_rc_body_pred)*100) )

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(rcf_body, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

xgb_headline = XGBClassifier()

xgb_headline.fit(X_headline_tfidf_train, y_headline_train)
y_xgb_headline_pred = xgb_headline.predict(X_headline_tfidf_test)

# print metrics
print ("XGBoost F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_xgb_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_xgb_headline_pred)*100) )

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)

cv = cross_validation(xgb_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

xgb_body = XGBClassifier()

xgb_body.fit(X_body_train_tfidf, y_body_train)
y_xgb_body_pred = xgb_body.predict(X_body_test_tfidf)

# print metrics
print ("XGBoost F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%``".format( f1_score(y_body_test, y_xgb_body_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_xgb_body_pred)*100) )

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(xgb_body, xtrain, ytrain , n_splits=5,init_chunk_size = 5000, chunk_spacings = 1000, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()