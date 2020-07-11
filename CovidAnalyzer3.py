import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
#IMPORTING DATASET

data=pd.read_csv("tweets10720.csv",engine="python")
num=len(data)

corpus = []
for i in range(0,num):
    rev=re.sub(r'http\S+', '',data['tweet'][i] )
    review = re.sub('[^a-zA-Z0-9]', ' ',rev)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    sw=stopwords.words('english')
    sw.remove('not')
    review = [ps.stem(word) for word in review if not word in set(sw) ]
    review = ' '.join(review)
    corpus.append(review)
data['tweet']=corpus
#CLASSIFICATION

i=999
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
X_train=[]
X_test=[]
y=[]
while i<num:
    y=data.iloc[:i,6].values
    data=pd.read_csv('tweets10720.csv',nrows=i+1000,engine="python")
    X_train=X[:i]
    X_test=X[i:i+1000]
    classifier = BernoulliNB()
    classifier.fit(X_train, y)
    y_pred = classifier.predict(X_test)
    y=y.tolist()
    if((num-i)>=1000):
        for k in range(1000):
            y.append(y_pred[k])
    else:
        for k in range(num-i):
            y.append(y_pred[k])   
    data['labels']=pd.Series(y)
    i+=1000
data=data[1000:] 
data.to_csv("corona10.csv")

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
    
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test, y_pred)
print (accuracy_score(y_test, y_pred))
