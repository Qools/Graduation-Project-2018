import zipfile
import os
import json
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# First file to start with
df = pd.read_json("/Users/asus/desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/tempdata/articles/2017-05-01.json")

# Combining all json files to a one data frame
count = 0
for filename in os.listdir("/Users/asus/desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/tempdata/news_of_interest/"):
    count+=1
    if count>2:
        file_path = "/Users/asus/desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/tempdata/articles/" + filename
        df_ = pd.read_json(file_path , encoding='uth-8')
        df = pd.concat(objs= [df,df_], axis=0,ignore_index=True)
        
        # To see the progress, print the number of the file
        if count%100 ==0:
            print("This is the ",count,"th file -->")

# Getting bodytext from the dataframe
lst = []
for i in range(df.shape[0]):
    lst.append(df.fields[i]["bodyText"])

# Creating a column with bodytext
df["bodyText"] = lst

# Getting headlines from the dataframe
lst_head = []
for i in range(df.shape[0]):
        lst_head.append(df.fields[i]["headline"])

# Creating a column with headline
df["headline"] = lst_head

# Getting rid of the embty bodies
df = df[df.bodyText != ""]

# Total Number of Articles
df.shape

# Filtering articles based on topics
df = df[(df.sectionName == 'US news') | (df.sectionName == 'Business') | (df.sectionName == 'Politics') | (df.sectionName == 'World news')]

# Getting to see the dataframe info (data type, non-null value etc.)
df.info()

# To see how many articles in each category
Counter(df.sectionName)

# Overview for the dataframe
df.head()

# Publication data range
df.webPublicationDate.min() ,df.webPublicationDate.max()

for idx,item in enumerate(df_test.bodyText):
    df_test.bodyText[idx] = re.sub('[^\x00-\x7F]+', "", item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

for idx,item in enumerate(df_test.headline):
    df_test.headline[idx] = re.sub('[^\x00-\x7F]+', "", item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

for idx, item in enumerate(df_test.bodyText):
    df_test.bodyText[idx] = re.sub('(\\n)',"",item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

for idx, item in enumerate(df_test.headline):
    df_test.headline[idx] = re.sub('(\\n)',"",item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

# Saving the cleaned data into csv file
df_test.to_csv("/Users/asus/desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/tempdata/Clean_TheGuardian_Combined_No_Slash.csv")

# Reading the cleaned data into data frame
df_test = pd.read_csv("/Users/asus/desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/tempdata/Clean_TheGuardian_Combined_No_Slash.csv")

df_test.info()

sum_ = 0
max_ = 0
min_ = 9999
for item in df.bodyText:
    sum_+=len(item)
    if len(item)>max_:
        max_ = len(item)
    if len(item)<min:
        min_ = len(item)

print("Min number of word count",min_ ,"\nMax number of word count",max_,"\nTotal number of word count", sum_)
