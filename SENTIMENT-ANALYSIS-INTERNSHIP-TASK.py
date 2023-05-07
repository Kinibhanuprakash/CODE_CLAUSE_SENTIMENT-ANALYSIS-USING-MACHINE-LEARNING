#!/usr/bin/env python
# coding: utf-8

# 
# ### Internship on "Data Science" at CodeClause
# ###### Project - SENTIMENT ANALYSIS

# Data Science Virtual Internship<br>
# Name:KINI BHANU PRAKASH REDDY<br>
# Date:7/5/2023

# ### Problem Statement

# Sentiment analysis, also refers as opinion mining, is a sub machine learning task where
# we want to determine which is the general sentiment of a given document. Using machine
# learning techniques and natural language processing we can extract the subjective information
# of a document and try to classify it according to its polarity such as positive, neutral or negative.<br>
# <br>
# It is a really useful analysis since we could possibly determine the overall opinion about a selling
# objects, or predict stock markets for a given company like, if most people think positive about it,
# possibly its stock markets will increase, and so on. Sentiment analysis is actually far from to be
# solved since the language is very complex (objectivity/subjectivity, negation, vocabulary,
# grammar,...) but it is also why it is very interesting to working on.

# ## <div id="toc">Table of Contents</div>
# <ol>
#     <li><a href=#ImportRequiredLibraries style="text-decoration:none">Import Required Libraries</a></li>
#     <li><a href=#LoadtheData style="text-decoration:none">Load the Dataset</a></li>
#     <li><a href=#EDA style="text-decoration:none">Exploratory Data Analysis (EDA)</a></li>
#     <li><a href=#ModelSelection style="text-decoration:none">Model Selection and Evaluations</a></li>
#     <li><a href=#AccuracyandClassificationReport style="text-decoration:none">Accuracy and Classification Report</a></li>

# #### <div id="ImportLibraries">1. Importing the required packages <a href="#toc" style="text-decoration:none">[ Top ]</a></div>

# In[1]:


import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# #### <div id="LoadDataset">2. Load Dataset <a href="#toc" style="text-decoration:none">[ Top ]</a></div>

# In[2]:


data=pd.read_csv("C:\\Users\\prakash\\Downloads\\test.csv",encoding= 'unicode_escape')


# In[3]:


data


# #### <div id="EDA">3. Exploratory Data Analysis (EDA) <a href="#toc" style="text-decoration:none">[ Top ]</a></div>

# In[4]:


data.isnull().sum()


# In[5]:


data.dropna(inplace=True)


# In[6]:


data.isnull().sum()


# In[7]:


data.shape


# In[8]:


data


# In[9]:


data["text"].values


# In[10]:


data=data.drop(data[["textID","Time of Tweet","Age of User","Country","Population -2020","Land Area (Km²)","Density (P/Km²)"]],axis=1)


# In[11]:


data.sentiment.value_counts()


# In[12]:


import nltk
nltk.download()


# #### <div id="ModelSelection">4. Model Selection and Evaluation<a href="#toc" style="text-decoration:none">[ Top ]</a></div>

# In[13]:


# preprocessing loop

import re
r
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


# In[14]:


corpus = []

for i in data['text']:
    r = re.sub('[^a-zA-Z]', '',i)
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)


# In[15]:


data["txt"]=corpus


# In[16]:


data


# In[17]:


data=data.drop(["text"],axis=1)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[19]:


x=data["txt"]
y=data["sentiment"]


# In[20]:


x=cv.fit_transform(x)


# In[21]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=12)


# In[23]:


model.fit(x_train,y_train)


# In[24]:


y_pred=model.predict(x_test)
y_pred


# In[25]:


x1="itsatamimverytiredbuticantsleepbutitryit"
x2=cv.transform([x1]).toarray()
model.predict(x2)


# In[26]:


x1="happybday"
x2=cv.transform([x1]).toarray()
model.predict(x2)


# #### <div id="AccuracyandClassificationReport">5.Accuracy and Classifiaction Report<a href="#toc" style="text-decoration:none">[ Top ]</a></div>

# In[27]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[28]:


from sklearn.metrics import classification_report


# In[29]:


print(classification_report(y_test,y_pred))


# In[ ]:




