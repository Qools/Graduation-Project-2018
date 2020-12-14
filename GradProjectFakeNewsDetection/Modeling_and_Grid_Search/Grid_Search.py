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
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.metrics import f1_score, accuracy_score


import os
import time
from sklearn.datasets import load_files
from sklearn.cross_validation import  ShuffleSplit
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import cross_val_score
#%matplotlib inline




df_all= pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/Complete_DataSet_Clean.csv")

### Assignig predictors and target values
X_body_text = df_all.body.values
X_headline_text = df_all.headline.values
y = df_all.fakeness.values



#ngram_range = [(1,1),(1,2),(1,3)]
ngram_range = [(1,1),(1,2)]

#max_df = [0.65,0.75,0.85,0.90]
#min_df = [0.001,0.01,0.1,0.25]
max_df = [0.75,0.85]
min_df = [0.01,0.1]

#penal = ['l1','l2']
penal = ['l1']

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

                #X_headline_train_tfidf, X_headline_test_tfidf, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
                X_body_train_tfidf, X_body_test_tfidf, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)

                lr = LogisticRegression(penalty=penalt,n_jobs=3)
                lr.fit(X_body_train_tfidf, y_body_train)
                y_pred = lr.predict(X_body_test_tfidf)

                print "Logistig Regression F1 and Accuracy Scores : \n"
                print "F1 score {:.4}%".format( f1_score(y_body_test, y_pred, average='macro')*100 ) 
                print "Accuracy score {:.4}%\n\n".format(accuracy_score(y_body_test, y_pred)*100 )
                
                f1_sc_lst.append(f1_score(y_body_test, y_pred, average='macro')*100)
                acc_lst.append(accuracy_score(y_body_test, y_pred)*100)

f1_sc_lst.sort()
acc_lst.sort()