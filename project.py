
import os, json 
import pandas as pd
  
path_to_json = 'C:/Users/Ankit Gokhroo/Desktop/fsdp/Day_07/json'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
 
json_files=json_files[:500]

df=pd.DataFrame()
for i in json_files:
    with open(str(i)) as json_file:
        json_text=json.load(json_file)
        if len(json_text['Reviews'])==0:
            pass
        else:
            df=df.append(json_text['Reviews'])
df=df.dropna()

features=df.iloc[:,[6]]
features.reset_index(inplace=True)
features=features.iloc[:,1:]

import nltk
from textblob import TextBlob       

#Testing NLP - Sentiment Analysis using TextBlob
TextBlob("The movie is good").sentiment

#Calculating the Sentiment Polarity
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

#Adding the Sentiment Polarity column to the data
features['polarity']=polarity
features['subjectivity']=subjectivity        

features['label']=' '

for i in range(len(features)):
    if features['polarity'][i]>0:
        features.at[i,'label']=1
    elif features['polarity'][i]<0:
        features.at[i,'label']=0
    else:
        features.at[i,'label']=2
         

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer  
corpus = []      

for i in range(len(features)):
    review = re.sub('[^a-zA-Z]', ' ', features['Title'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

features['corpus']=corpus

from sklearn.feature_extraction.text import TfidfVectorizer 
tf_idf = TfidfVectorizer(max_features=50000)
tf_data = tf_idf.fit_transform(corpus)
t=tf_data.toarray()
labels = features['label'].values.astype('float64')
 
# naiver bayes mutinomial 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(t, labels, test_size = 0.20, random_state = 0)

#naive bayes bernoulii
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
classifier =BernoulliNB()
classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import accuracy_score 
print (accuracy_score(labels_test, labels_pred)) #can be calcuated from cm as well

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_bnb = confusion_matrix(labels_test, labels_pred)

score_train_bnb = classifier.score(features_train, labels_train)
score_test_bnb = classifier.score(features_test, labels_test)


features["label"].value_counts(normalize=True)

#sampling of the data
neg_review=len(features[features['label']==0])
sample_data = features[features['label']==1].sample(neg_review, replace=True)
neg_data = features[features['label']==0]

over_sample_df = pd.concat([sample_data,neg_data], axis=0)
over_sample_class_counts=pd.value_counts(over_sample_df['label'])


over_sample_class_counts.plot(kind='bar')
plt.xlabel = 'label' 
plt.ylabel = 'Frequency'



from sklearn.feature_extraction.text import TfidfVectorizer 
tf_idf = TfidfVectorizer(max_features=50000)
tf_data = tf_idf.fit_transform(list(over_sample_df['corpus']))
t=tf_data.toarray()

obj_dict={}
obj_dict["Vectorizer"]=tf_idf

labels = over_sample_df['label'].values.astype('float64')
 
# naiver bayes mutinomial 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(t, labels, test_size = 0.20, random_state = 0)


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
classifier = BernoulliNB()
classifier.fit(features_train, labels_train)

obj_dict["BNB"]=classifier 
# Predicting the Test set results
labels_pred = classifier.predict(features_test)


from sklearn.metrics import accuracy_score 
print (accuracy_score(labels_test, labels_pred)) #can be calcuated from cm as well

corpus1=[]
review  ="hotel is awesome"
#perform row wise noise removal and stemming

#let's do it on just first row data
review = re.sub('[^a-zA-Z]', ' ', review)
review = review.lower()
review = review.split()

review = [word for word in review if not word in set(stopwords.words('english'))]
    
ps = PorterStemmer()
review = [ps.stem(word) for word in review]

review = ' '.join(review)
corpus1.append(review)
#lem = WordNetLemmatizer() #Another way of finding root word
#review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]

review=tf_idf.transform(corpus1)

classifier.predict(review)

import pickle
pickle_out=open("C:/Users/Ankit Gokhroo/Desktop/web/static/models.pkl","wb")
pickle.dump(obj_dict,pickle_out)
pickle_out.close()


