import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

df1 = pd.read_csv("/Users/Yemre/Desktop/CapStone/NYT_Articles_.csv")
df2 = pd.read_csv("/Users/Yemre/Desktop/NYT_Article_Collections/New_Times_Last_SON_aprl_3.csv")
df3 = pd.read_csv("/Users/Yemre/Desktop/NYT_Article_Collections/New_Times_Last_SON_aprl_3_2nd.csv")
df4 = pd.read_csv("/Users/Yemre/Desktop/NYT_Article_Collections/New_Times_Last_SON_aprl_4.csv")


df_all = pd.concat([df1,df2,df3,df4],axis=0,ignore_index=True).reset_index()

# Freeing some space
del df1,df2,df3,df4

# Number of articles in each section
Counter(df_all.section_name)

# Getting only wanted sections
df_all = df_all[(df_all.section_name == 'World') | (df_all.section_name == 'U.S.') | (df_all.section_name == 'Business Day')]

df_all.shape

for idx, item in enumerate(df_all.body):
    df_all.body[idx] = re.sub('[^\x00-\x7F]+', "", item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

df_all.info()

# Dropping unnecesary columns
df_all = df_all.drop(["index"],axis=1)

df_all = df_all.reset_index()

df_all.headline[29]

# Dropping all the unwated columns
df_all = df_all.drop(["abstract","blog","byline","document_type","multimedia","news_desk","print_page","slideshow_credits","snippet","subsection_name","type_of_material","web_url","source","Unnamed: 0"],axis=1)

# Getting only needed main headline
lst_head = []
for idx,item in enumerate(df_all.headline): 
    lst_head.append(" ".join(df_all.headline[idx].replace("'","").split(",")[0].split(" ")[1:]))

df_all = df_all.drop(["headline"],axis=1)

df_all['head_clean'] = lst_head

df_all.dropna(inplace=True)

df_all.info()

# Cleaning the data
for idx, item in enumerate(df_all.body):
    df_all.body[idx] = re.sub('(\\n)',"",item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

# Cleaning the data
for idx, item in enumerate(df_all.body):
    df_all.body[idx] = re.sub('(\\n)',"",item)
    if idx%500 == 0:
        print('Here is the ',idx,'th item')

#Cleaning the data

slash_text = []
slash_title = []
for idx, item in enumerate(df_all.body):
    
    try:
        df_all.body[idx] = re.sub('(\\n)',"",item)
    except:
        df_all.body[idx] = 'Dummy_Text'
        wrong_text.append((idx,item))
    
    if idx%500 == 0:
        print('Here is the ',idx,'th item')
        
        
for idx, item in enumerate(df_all.head_clean):
    
    try:
        df_all.head_clean[idx] = re.sub('(\\n)',"",item)

    except:
        df_all.head_clean[idx] = 'Dummy_Title'
        wrong_title.append((idx,item))
    if idx%500 == 0:
        print('Here is the ',idx,'th item')


# Saving the clean data to csv file
df_all.to_csv("/Users/Yemre/Desktop/NYT_Combined_Clean_APR_4_No_Slash.csv")

tdf = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
vectorizer = tdf.fit(df_all.body)
transformed_text = vectorizer.transform(df_all.body)
transformed_title = vectorizer.transform(df_all.head_clean)

def get_tfidf_term_scores(feature_names):
    '''Returns dictionary with term names and total tfidf scores for all terms in corpus'''
    term_corpus_dict = {}
    # iterate through term index and term 
    for term_ind, term in enumerate(feature_names):
        term_name = feature_names[term_ind]
        term_corpus_dict[term_name] = np.sum(transformed_title.T[term_ind].toarray())
        
    return term_corpus_dict

# list of features created by tfidf
feature_names = tdf.get_feature_names()

term_corpus_dict = get_tfidf_term_scores(feature_names)

print("Number of columns is: ",len(term_corpus_dict.keys()))

def get_sorted_tfidf_scores(term_corpus_dict):
    '''Returns sort words from highest score to lowest score'''
    # sort indices from words wit highest score to lowest score
    sortedIndices = np.argsort( list(term_corpus_dict.values()))[::-1]
    # move words and score out of dicts and into arrays
    termNames = np.array(list(term_corpus_dict.keys()))
    scores = np.array(list(term_corpus_dict.values()))
    # sort words and scores
    termNames = termNames[sortedIndices]
    scores = scores[sortedIndices]
    
    return termNames, scores

termNames, scores = get_sorted_tfidf_scores(term_corpus_dict)

def getSelectScores(selectTerms):
    '''Returns a list of tfidf scores for select terms that are passed in'''
    score = [ term_corpus_dict[select_term]  for select_term in selectTerms]
    return score

selectTerms = ['trump', 'clinton','islamic', 'russia' , 'women', 'obama', 'men',
               'students', 'shooting', 'democrats', 'republicans', 'climate',
               'education', 'environment', 'tech', 'minorities', 'carbon',
               'muslim','ban']

selectScores = getSelectScores(selectTerms)

def plot_tfidf_scores(scores,termNames, selectScores, selectTerms,  n_words = 18):
    '''Returns one plot for Importance of Top N Terms
       and one plot for Importance of Select K Terms'''

    # Create a figure instance, and the two subplots
    fig = plt.figure(figsize = (14, 18))
    
    override = {'fontsize': 'large'}

    fig.add_subplot(221)   #top left
    n_words = 75
    sb.barplot(x = scores[:n_words], y = termNames[:n_words]);
    sb.plt.title("TFIDF - Importance of Top {0} Terms".format(n_words));
    sb.plt.xlabel("TFIDF Score");

    fig.add_subplot(222)   #top right 
    sb.barplot(x = selectScores, y = selectTerms);
    sb.plt.title("TFIDF - Importance of Select {0} Terms".format(len(selectTerms)));
    sb.plt.xlabel("TFIDF Score");
    sb.plt.ylabel(override)

plot_tfidf_scores(scores, termNames, selectScores, selectTerms,  n_words = 18)