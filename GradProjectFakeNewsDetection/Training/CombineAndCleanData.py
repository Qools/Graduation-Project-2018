#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

# Reading NYT dataset
df_nyt = pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/NYTDataset/NYT_Combined_Clean_APR_4_No_Slash.csv")

# Adding label to real news to the dataset and showing the column names
df_nyt["fakeness"] = 0
print df_nyt.columns

# Reading The Guardian dataset
df_tguard = pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/GuardianDataset/Clean_TheGuardian_Combined_No_Slash.csv")

# Adding label to real news to the dataset and showing the column names
df_tguard["fakeness"] = 0
print df_tguard.columns

df_fake = pd.read_csv("/Users/asus/source/repos/Detecting_Fake_News-master/fakeDataset/FakeNews_Clean_All.csv")

# Label was there for fake news dataset and showing the column names
print df_fake.columns

# Changing the name of the column for concating later
df_tguard = df_tguard.rename(columns={'bodyText' : 'body','webPublicationDate':'pub_date'})
df_nyt = df_nyt.rename(columns={'head_clean':'headline', '_id':'id'})
df_fake = df_fake.rename(columns={'text':'body','title':'headline','uuid':'id','published':'pub_date'})
print df_fake.columns,df_nyt.columns,df_tguard.columns

# Dropping unnecesary columns
df_fake.drop([u'Unnamed: 0', u'ord_in_thread', u'language', u'crawled',  u'site_url',
             u'country', u'thread_title', u'spam_score', u'replies_count', u'participants_count', 
             u'likes', u'comments', u'shares', u'type'],inplace=True,axis=1)

df_nyt.drop([u'Unnamed: 0', u'word_count'],inplace=True,axis=1)

df_tguard.drop([u'Unnamed: 0', u'apiUrl', u'fields', u'isHosted',u'pillarId', u'pillarName', u'sectionId', 
                u'sectionName', u'type', u'webTitle', u'webUrl'],inplace=True,axis=1)

# Overview for data
print df_fake.head()
# Overview for data
print df_nyt.head()

# Overview for data
print df_tguard.head()

# Concat the datasents
df_all = df_fake.append(df_tguard, ignore_index=True)
df_all = df_all.append(df_nyt,ignore_index=True)

#Dropping the Nan values and info
df_all.dropna(inplace=True)
print(df_all.shape)
print df_all.info()

print df_all.head()

df_all.to_csv("/Users/asus/source/repos/Detecting_Fake_News-master/Complete_DataSet_Clean.csv")
