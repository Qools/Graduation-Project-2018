import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sb
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# Reading the data set that has been downloaded from Kaggle.com
df  = pd.read_csv('/Users/asus/Desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/Cleaning_The_Data/fake.csv')

# Information about the data
df.info()

#Overview of the dataset
df.head()

df.language.unique()

# Pablication dates
print("Lates: ",df.published.max(),"\n\nEarliest",df.published.min())

US_corpus = df.text.values

df.shape

# Replacing empty articles with "Dummy_text and Dummy_Title"
wrong_text = []
wrong_title = []
for idx, item in enumerate(df.text):
    
    try:
        df.text[idx] = re.sub('[^\x00-\x7F]+', "", item)
    except:
        df.text[idx] = 'Dummy_Text'
        wrong_text.append((idx,item))
    
    if idx%500 == 0:
        print('Here is the ',idx,'th item')
        
        
for idx,item in enumerate(df.title):
    
    try:
        df.title[idx] = re.sub('[^\x00-\x7F]+', "", item)
    except:
        df.title[idx] = 'Dummy_Title'
        wrong_title.append((idx,item))
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

slash_text = []
slash_title = []
for idx, item in enumerate(df.text):
    
    try:
        df.text[idx] = re.sub('(\\n)',"",item)
    except:
        df.text[idx] = 'Dummy_Text'
        wrong_text.append((idx,item))
    
    if idx%500 == 0:
        print('Here is the ',idx,'th item')
        
        
for idx, item in enumerate(df.title):
    
    try:
        df.title[idx] = re.sub('(\\n)',"",item)

    except:
        df.title[idx] = 'Dummy_Title'
        wrong_title.append((idx,item))
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

# Lookin at the data to see neccesary columns
df.info()

# Dropping Unnecassary Columns
df = df.drop(['domain_rank','main_img_url','Unnamed: 0','author'],axis=1)

# Dropping Nan values
df.dropna(inplace=True)

#After Cleaning the data
df.info()

# Adding label to the data
df['fakeness'] = 1

# Saving it for the future use
df.to_csv('/Users/asus/Desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/Cleaning_The_Data/FakeNews_Clean_All.csv')

# Importing the dataset
df = pd.read_csv('/Users/asus/Desktop/Class 4/2/Graduation Project II/GradProjectFakeNewsDetection/GradProjectFakeNewsDetection/Cleaning_The_Data/FakeNews_Clean_All.csv')

# Double check to see if is clean 
df.info()

df

# Plotting the word count and importance
plot_tfidf_scores(scores, termNames, selectScores, selectTerms,  n_words = 18)
