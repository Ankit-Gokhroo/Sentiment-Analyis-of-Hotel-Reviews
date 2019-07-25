# -- coding: utf-8 --
"""
Created on Sun Jul 14 21:15:24 2019

@author: Ankit Gokhroo
""" 

#importing  packages
import os, json 
import pandas as pd
from matplotlib import pyplot as plt 
import re
#NLP packages
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from textblob import TextBlob   

#path of dataset where data is store in json format  
path_to_json = 'C:/Users/Ankit Gokhroo/Desktop/fsdp/Day_07/Project/json'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
json_files=json_files[:500]

#making new dataframe
df=pd.DataFrame()

#load data from json files and fit it in dataframe
for i in json_files:
    with open(str(i)) as json_file:
        json_text=json.load(json_file)
        if len(json_text['Reviews'])==0:
            pass
        else:
            df=df.append(json_text['Reviews'])

#dropping nan values
df=df.dropna()

#selecting Title column as a feature in a dataframe
features=df.iloc[:,[6]]
features.reset_index(inplace=True)  
features=features.iloc[:,1:]
features=features.reset_index()
   
#Calculating the Sentiment Polarity and Subjectivity
polarity=[] # list which will contain the polarity of the comments 
subjectivity=[] # list which will contain the subjectivity of the comments

for i in features['Title'].values:
    try:
        analysis =TextBlob(i)
        polarity.append(analysis.sentiment.polarity)
        subjectivity.append(analysis.sentiment.subjectivity)
        
    except:
        polarity.append(0)
        subjectivity.append(0)
        
#function returning common words according to polarity and subjectivity      
def wc(data,bgcolor,title):
    plt.figure(figsize = (10,10))
    wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 50)
    wc.generate(' '.join(data)) 
    plt.imshow(wc)
    plt.axis('off')         
        
#Adding the Sentiment Polarity column to the data
features['polarity']=polarity
features['subjectivity']=subjectivity        
        

#polarity < 0 depicts negative sentiment
#polarity > 0 depicts positive sentiment
#polarity = 0 depicts neutral sentiment
df_negative=features[['Title','polarity','subjectivity']][features.polarity<0]
df_positive=features[['Title','polarity','subjectivity']][features.polarity>0]           
df_neutral=features[['Title','polarity','subjectivity']][features.polarity==0]                 
 
wc(features['Title'][features.polarity==0],'black','Common Words' )  
 
a_indexof_df_positive=df_positive.index.tolist()
b_indexof_df_negative=df_negative.index.tolist()
c_indexof_df_neutral=df_neutral.index.tolist()

#adding 'label' column
features['label']=' '

#making of labels in dataframe
for i in range(len(features)):
    if features.index[i] in a_indexof_df_positive:
        features.at[i,'label']=1
    elif features.index[i] in b_indexof_df_negative:
        features.at[i,'label']=0
    else:
        features.at[i,'label']=2
                 
#extracting important words from the given review         
corpus = []
 
for i in range(len(features)):
    review = re.sub('[^a-zA-Z]', ' ',features['Title'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#adding corpus column in features dataframe
features['corpus']=corpus

#encoding of features by term frequency method
from sklearn.feature_extraction.text import TfidfVectorizer 
tf_idf = TfidfVectorizer(max_features=15000)
tf_data = tf_idf.fit_transform(corpus)
t=tf_data.toarray()   #features
labels = features.iloc[:,4].values.astype('float64')
                        
#visulaisation of reviews       
count=features['label'].value_counts()
count.plot(kind='bar')

#sampling of data for better predictions
fraud_sample = features[features['label']==1].sample(len(features[features['label']==0]), replace=True)
df_fraud = features[features['label']==0]


over_sample_df = pd.concat([fraud_sample,df_fraud], axis=0)
over_sample_class_counts=pd.value_counts(over_sample_df['label'])

#visualisation of reviews after sampling 
over_sample_class_counts.plot(kind='bar')
plt.xlabel = 'label' 
plt.ylabel = 'Frequency'

#encoding of final new features
from sklearn.feature_extraction.text import TfidfVectorizer 
tf_idf = TfidfVectorizer(max_features=15000)
tf_data = tf_idf.fit_transform(list(over_sample_df['corpus']))
t=tf_data.toarray()   #features for model
labels = over_sample_df.iloc[:,4].values.astype('float64')   #labels for model
  
#splitting features and labels into training and testing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(t, labels, test_size = 0.20, random_state = 0)

#from sklearn applying naive bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
classifier = BernoulliNB()

#fitting the model
classifier.fit(features_train, labels_train)
  
# Predicting the Test set results
labels_pred = classifier.predict(features_test)
 
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)
  
#training and testing score
training_score=classifier.score(features_train,labels_train)
testing_score=classifier.score(features_test,labels_test)

#testing prediction

x='you nyc dish '
x = re.sub('[^a-zA-Z]', ' ',x)
x = x.lower()
x = x.split()

x = [word for word in x 
          if not word 
          in set(stopwords.words('english'))]


corpus1=[]    
#lem = WordNetLemmatizer() #Another way of finding root word
ps = PorterStemmer()
x = [ps.stem(word) for word in x]
#review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
x = ' '.join(x)
corpus1.append(x)

t= tf_idf.transform(corpus1)
t=t.toarray()   #features

classifier.predict(t)

#visualisations
#Displaying the POSITIVE comments
features[['Title','polarity','subjectivity']][features.polarity>0].head(10)

#Displaying the NEGATIVE comments
features[['Title','polarity','subjectivity']][features.polarity<0].head(10)

#Displaying highly subjective reviews
features[['Title','polarity','subjectivity']][features.subjectivity>0.8].head(10)
wc(features['Title'][features.subjectivity>0.8],'black','Common Words' )

#Displaying highly positive reviews
features[['Title','polarity','subjectivity']][features.polarity>0.8].head(10)
wc(features['Title'][features.polarity>0.8],'black','Common Words' )

#Displaying highly negative reviews
features[['Title','polarity','subjectivity']][features.polarity<-0.4].head(10)
wc(features['Title'][features.polarity<-0.4],'black','Common Words' )
 
features.polarity.hist(bins=50)
features.subjectivity.hist(bins=50)

over_sample_df.polarity.hist(bins=50)
over_sample_df.subjectivity.hist(bins=50)

# plot sentiment distribution for positive and negative reviews

import seaborn as sns

for x in [0, 1,2]:
    subset = features[features['label'] == x]
    
    # Draw the density plot
    if x == 1:
        label = "Good reviews"
    elif x==0:
        label = "Bad reviews"
    else:
        label="Neutral reviews"
    sns.distplot(subset['polarity'], hist = False, label = label)
     
    
for x in [0, 1,2]:
    subset = features[features['label'] == x]
    
    # Draw the density plot
    if x == 1:
        label = "Good reviews"
    elif x==0:
        label = "Bad reviews"
    else:
        label="Neutral reviews"
    sns.distplot(subset['subjectivity'], hist = False, label = label)    
    
    
for x in [0, 1]:
    subset = over_sample_df[over_sample_df['label'] == x]
    
    # Draw the density plot
    if x == 1:
        label = "Good reviews"
    else:
        label="Bad reviews"
    sns.distplot(subset['polarity'], hist = False, label = label)    
    
from sklearn.metrics import classification_report

print (classification_report(labels_test, labels_pred, target_names=['NO', 'YES']))