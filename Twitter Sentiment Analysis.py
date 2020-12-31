#!/usr/bin/env python
# coding: utf-8

# In[1]:


#download libraries
import re
import nltk
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth",200)
warnings.filterwarnings("ignore",category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read train and test dataset
train=pd.read_csv("C:/Users/hp/Downloads/Twitter_Sentiment_Analysis/train.csv")
test=pd.read_csv("C:/Users/hp/Downloads/Twitter_Sentiment_Analysis/test.csv")


# # Data Inspection

# In[ ]:





# In[3]:


# check some non racist/sexiest twweets
train[train['label']==0].head(5)


# In[4]:


# check some racist/sexiest twweets
train[train['label']==1].head(5)


# In[5]:


#dimensions of the train and test datasets
train.shape,test.shape


# In[6]:


# label disticution at the train dataset
train["label"].value_counts()


# In[7]:


# let's check the distribution of length of tweets in terms of words in train and test datasets
length_tr= train['tweet'].str.len()
length_te= test['tweet'].str.len()
plt.hist(length_tr,bins=20,label="train tweets")
plt.hist(length_te,bins=20,label="test tweets")
plt.legend()
plt.show()


# # Data cleaning

# In[8]:


#combine the training and testing datasets
combine= train.append(test, ignore_index=True)
combine.shape


# In[9]:


# remove unwanted text patterns from the tweets
def remove(text_input,pattern):
    r=re.findall(pattern,text_input)
    for i in r:
        text_input=re.sub(i,'',text_input)
    return text_input


# In[10]:


# remove twitter handlers(@user)
# we create a column containing the cleaned tweets
combine['cleaned_tweets']=np.vectorize(remove)(combine['tweet'],"@[\w]*")
combine.head(5)


# In[11]:


# remove punctuations, numbers, and special Characters
#replace eveything  except characters and hashtags with spaces
combine['cleaned_tweets']=combine['cleaned_tweets'].str.replace("[^a-zA-Z#]"," ")
combine.head(5)


# In[12]:


#remove short words
#remove all the words having length 3 or less
combine['cleaned_tweets']=combine['cleaned_tweets'].apply(lambda x:' '.join([j for j in x.split() if len(j)>3]))
combine.head(5)


# In[13]:


#Tokenization
tokenize_tweets=combine['cleaned_tweets'].apply(lambda x: x.split())
tokenize_tweets.head(5)


# In[14]:


# normalize the tokenized tweets
from nltk.stem.porter import*
#Stemming
stemmer=PorterStemmer()
tokenize_tweets=tokenize_tweets.apply(lambda x: [stemmer.stem(i) for i in x])
tokenize_tweets.head(5)


# In[15]:


#let's stitch these tokens back together
for i in range(len(tokenize_tweets)):
    tokenize_tweets[i]=' '.join(tokenize_tweets[i])
combine['cleaned_tweets']=tokenize_tweets


# In[16]:


combine.head(5)


# # Tweets visualization

# In[17]:


# visualizing all the words our data using WordCloud plot
from wordcloud import WordCloud
words=' '.join([text for text in combine['cleaned_tweets']])
wordcloud=WordCloud(width=800,height=500,random_state=20,max_font_size=110).generate(words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show

We can observe that the most of words are positif or neutral like love,great,thank, and life which are the most frequent ones. Still, we don't get any idea about the words associated with racist or sexist tweets.
# In[18]:


#separate wordclouds for both classes.
# words in non racist/sexist tweets
simple_tweets=' '.join([text for text in combine['cleaned_tweets'][combine['label']==0]])
wordcloud=WordCloud(width=800,height=500,random_state=20,max_font_size=110).generate(simple_tweets)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show


# In[19]:


# words in racist/sexist tweets
negative_tweets=' '.join([text for text in combine['cleaned_tweets'][combine['label']==1]])
wordcloud=WordCloud(width=800,height=500,random_state=20,max_font_size=110).generate(negative_tweets)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show


# In[20]:


#undersatnd the impact of hashtags on tweets sentiment
# create a function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


# In[21]:


# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combine['cleaned_tweets'][combine['label'] == 0])
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combine['cleaned_tweets'][combine['label'] == 1])
# unnest the list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[22]:


# plot the top hashtags of simple tweets
aa = nltk.FreqDist(HT_regular)
bb = pd.DataFrame({'Hashtags': list(aa.keys()),
                  'Count': list(aa.values())})
# selecting top 20 of the most frequent hashtags     
bb = bb.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=bb, x= "Hashtags", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[23]:


# plot the top hashtags of racist/sexist tweets
cc = nltk.FreqDist(HT_negative)
dd = pd.DataFrame({'Hashtags': list(cc.keys()), 
                  'Count': list(cc.values())})
# selecting top 20 of the most frequent hashtags
dd = dd.nlargest(columns="Count", n = 20)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=dd, x= "Hashtags", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

The most of the terms are negative with a few neutral terms. So, we can keep these hashtags in our data as they contain useful information. Next, we will try to extract features from the tokenized tweets.
# # Extracting Features from Cleaned Tweets

# In[24]:


# using BOW
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')#set the parameter max_features = 1000 to select only top 1000 terms ordered by term frequency across the corpus.
# bag-of-words feature matrix
BOW = vectorizer.fit_transform(combine['cleaned_tweets'])
vectorizer,BOW


# In[25]:


# Using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combine['cleaned_tweets'])
tfidf_vectorizer,tfidf


# In[26]:


# Using words embeddings
#Word2Vec Features
import gensim
from gensim.models import Word2Vec
tokenize_tweets=combine['cleaned_tweets'].apply(lambda x: x.split())
model=gensim.models.Word2Vec(tokenize_tweets,size=200,
                             window=5,
                             min_count=2,
                             sg=1,
                             hs=0,
                             negative=10,
                             workers=2,
                             seed=34)
model.train(tokenize_tweets,total_examples=len(combine['cleaned_tweets']), epochs=20)


# In[27]:


#the most similar words from the corpus
model.most_similar(positive="life")


# In[28]:


model.most_similar(positive="love")


# In[29]:


model.most_similar(positive="trump")


# In[30]:


# check the representation of a vector of some words in data
model['love'],len(model['love'])


# In[31]:


model['life'],len(model['life']) #the length of the vector


# In[32]:


# Preparing vectors for tweets
# create a function to create a vector for each tweet by taking the average of vectors
def vector_w(tokens,size):
    vec=np.zeros(size).reshape((1,size))
    count=0
    for word in tokens:
        try:
            vec+= model[word].reshape((1,size))
            count+= 1
        except KeyError:  #handling the case where the token isn't in vocabulary 
                        continue
    if count!= 0:
          vec/=count
    return vec


# In[33]:


#Preparing Word2Vec feature set
word_array=np.zeros((len(tokenize_tweets),200))
for i in range(len(tokenize_tweets)):
    word_array[i,:]= vector_w(tokenize_tweets[i],200)
    word_df= pd.DataFrame(word_array) 


# In[34]:


word_df.shape

We have 200 features while we had 1000 features in the BOW and TF-IDF
# # Modeling

# In[35]:


# using Random Forest algorithm
#using BOW features
# extracting train and test bow features
from sklearn.model_selection import train_test_split
train_b=BOW[:31962,:]
test_b=BOW[31962:,:]
#splitting data into training and validation set
xtrain,xvalid,ytrain,yvalid=train_test_split(train_b,train['label'],random_state=42,test_size=0.3)


# In[36]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
RF=RandomForestClassifier(n_estimators=400,random_state=11).fit(xtrain,ytrain)
prediction=RF.predict(xvalid)
#validation score 
f1_score(yvalid,prediction),accuracy_score(yvalid,prediction)


# In[38]:


# making prediction for the test dataset
test_p=RF.predict(test_b)
test['label']=test_p
submission= test[['id','label']]
submission.to_csv('submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




