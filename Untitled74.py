#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import pandas as pd
from transformers import pipeline
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sn
from nltk.corpus import stopwords


# In[2]:


files = [pd.read_csv('combine1.csv'),pd.read_csv('combine2.csv'),pd.read_csv('combine3.csv'),pd.read_csv('combine4.csv'),pd.read_csv('combine5.csv'),pd.read_csv('combine6.csv'), pd.read_csv('combine7.csv')]
df = pd.concat(files,ignore_index=True)
df


# In[3]:


df.to_csv("complete dataset3.csv")
df


# In[4]:


df = df.drop('Location', axis='columns')
df = df.drop('Name', axis='columns')
df = df.drop('User ID', axis='columns')
df = df.drop('Language', axis='columns')
df = df.drop('Default Profile', axis='columns')
df = df.drop('default Profile Image', axis='columns')
df = df.drop('description', axis='columns')
df = df.drop('favourites Count', axis='columns')
df = df.drop('follow', axis='columns')
df = df.drop('followersCount', axis='columns')
df = df.drop('following', axis='columns')
df = df.drop('friends', axis='columns')
df = df.drop('profile Background', axis='columns')
df = df.drop('profile image url', axis='columns')
df = df.drop('staruses Count', axis='columns')
df = df.drop('Time', axis='columns')
df


# In[5]:


df = df.drop_duplicates(keep=False)
df


# In[6]:


df = df.reset_index()


# In[7]:


df = df.drop('index', axis='columns')


# In[8]:


df = df.drop('Unnamed: 0', axis='columns')
df = df.drop('Unnamed: 0.1', axis='columns')
df


# In[9]:


def cleanTxt(text):
    text = text.lower() #converting whole text into lower
    text = re.compile('rt @').sub('@', text) # removing tags
    text= re.sub('@[A-Za-z0-9]+','',text ) #removing mentions
    text= re.sub("#",'',text) #removing mentions
    text = re.sub('[()!?]', ' ', text) #removing special letters
    text = re.sub(r'http\S+', '', text) #removing links
    text = re.sub('\[.*?\]',' ', text) #removing special letters
    text = re.sub("[^a-z0-9]"," ", text) #removing special letters
    text = re.compile('https?:\/\/\S+').sub('@', text, count=1) #removing links
    return text #to get an output of text after cleaning it

df['Tweets']= df['Tweets'].apply(cleanTxt)


# In[10]:


df = df.drop_duplicates(keep="last")
df = df.reset_index()
df = df.drop('index', axis='columns')
df


# In[11]:


df = df.reset_index()
df


# In[12]:


df = df.drop('index', axis='columns')


# In[13]:


df


# In[14]:



from nltk.corpus import stopwords 
stop = set(stopwords.words('english')) 
from nltk.corpus import stopwords 
def remove_stopword(word): 
   return word not in words 

df['Tweets'] = df['Tweets'].str.lower().str.split() 
df['Tweets'] = df['Tweets'].apply(lambda x : [item for item in x if item not in stop]) 


# In[15]:


df


# In[16]:


df['Tweet'] = [','.join(map(str, l)) for l in df['Tweets']]
df


# In[17]:


df = df.drop('Tweets', axis='columns')
df


# In[18]:


def cleanTxt(text):
    text = text.lower()
    text = re.compile('rt @').sub('@', text)
    text= re.sub('@[A-Za-z0-9]+','',text ) #removing mentions
    text= re.sub("#",'',text) #removing #
    text = re.sub('[()!?]', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('\[.*?\]',' ', text)
    text = re.sub("[^a-z0-9]"," ", text)
    text = re.compile('https?:\/\/\S+').sub('@', text, count=1)
    return text


df['Tweet']= df['Tweet'].apply(cleanTxt)
df


# In[19]:



df['Tweet']= df['Tweet'].apply(cleanTxt)

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# create a function to get the Polarity of all the tweets
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# create 2 columns 'Subjectivity' and 'Polarity'
df['Subjectivity']= df['Tweet'].apply(getSubjectivity)
df['Polarity']= df['Tweet'].apply(getPolarity)

def getAnalysis(score):
    if score<-0.2:
        return '0'
    else:
        return '1'

df['Analysis']= df['Polarity'].apply(getAnalysis)
df.to_csv('sen.csv')


# In[20]:


df = df.drop('Subjectivity', axis='columns')
df = df.drop('Polarity', axis='columns')


# In[21]:


df


# In[ ]:


negative_tweets = df[df['Analysis'] == '0']['Tweet'].tolist()
aug = naw.SynonymAug(aug_src='wordnet')
augmented_tweets = [aug.augment(tweet) for tweet in negative_tweets]
augmented_df = pd.DataFrame({'Tweet': augmented_tweets, 'analysis': '1'})


# In[ ]:


balanced_df = pd.concat([df[df['analysis'] == 'Positive'], augmented_df])


# In[ ]:


translator = pipeline('translation', model='akhooli/cambrideng-arabic')
df['Arabic_Tweet'] = df['Tweet'].apply(lambda x: translator(x, target_language='ar')[0]['translation_text'])


# In[ ]:


tweets = df['Arabic_Tweet'].apply(lambda x: x.split())


# In[ ]:


model = Word2Vec(tweets, min_count=1)


# In[ ]:


X = []
for tweet in tweets:
    embedding = []
    for word in tweet:
        if word in model.wv.vocab:
            embedding.append(model[word])
    X.append(embedding)


# In[ ]:


model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape=(100, 1)))
model_lstm.add(Dense(1, activation='sigmoid'))


# In[ ]:


model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = df['Analysis'][:train_size], df['Analysis'][train_size:]


# In[ ]:


X_train = X_train.reshape((X_train.shape[0], 100, 1))
X_test = X_test.reshape((X_test.shape[0], 100, 1))


# In[ ]:


model_lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# In[ ]:


# Make predictions on test data
y_pred = model_lstm.predict_classes(X_test)


# In[ ]:


conf_mat = confusion_matrix(y_test, y_pred)


# In[ ]:


print('Confusion matrix:')
print(conf_mat)


# In[ ]:


class_rep = classification_report(y_test, y_pred)


# In[ ]:


print('Classification report:')
print(class_rep)


# In[ ]:


]plt.plot(history.history['lr'])
plt.title('Model Learning Rate')
plt.ylabel('Learning rate')
plt.xlabel('Epoch')
plt.show()

