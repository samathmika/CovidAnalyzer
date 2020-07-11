#COVID ANALYZER

#Importing libraries

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
import stats
#Importing stats.dataset

X=stats.dataset.iloc[:,4].values
num=len(stats.dataset)
#Filling missing stats.dataset

def isNaN(string):
    return string != string

for i in range(num):
    if(isNaN(X[i])):
        X[i]="coronavirus"
stats.dataset['text']=X

#Cleaning the texts

corpus = []
for i in range(0, num):
    rev=re.sub(r'http\S+', '',stats.dataset['text'][i] )
    review = re.sub('[^a-zA-Z0-9]', ' ',rev)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    sw=stopwords.words('english')
    sw.remove('not')
    review = [ps.stem(word) for word in review if not word in set(sw) ]
    review = ' '.join(review)
    corpus.append(review)


#Obtaining sentiment scores
    
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
stats.dataset['neg']=stats.dataset['text'].apply(lambda x:sia.polarity_scores(x)['neg'])
stats.dataset['neu']=stats.dataset['text'].apply(lambda x:sia.polarity_scores(x)['neu'])
stats.dataset['pos']=stats.dataset['text'].apply(lambda x:sia.polarity_scores(x)['pos'])
stats.dataset['compound']=stats.dataset['text'].apply(lambda x:sia.polarity_scores(x)['compound'])

#Percentage values of sentiments

pos_review=[j for i,j in enumerate(stats.dataset['text']) if stats.dataset['compound'][i]>0.2]
neu_review=[j for i,j in enumerate(stats.dataset['text']) if 0.2>=stats.dataset['compound'][i]>=-0.2]
neg_review=[j for i,j in enumerate(stats.dataset['text']) if stats.dataset['compound'][i]<-0.2]
posp=format(len(pos_review)*100/len(stats.dataset['text']))
neup=format(len(neu_review)*100/len(stats.dataset['text']))
negp=format(len(neg_review)*100/len(stats.dataset['text']))

#CLUSTERING USING LENGTH

#Finding length of tweets

length=[]
for i in range(num):
    length.append(len(str(X[i])))
stats.dataset['len']=length

#Elbow method 

'''
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''

#K-Means for Clustering using length

Y=stats.dataset.iloc[:,[17,16]].values
kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)
fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 10, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 10, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 30, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')
'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Length of tweet')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()
fig.savefig("../flask/static/images/clusterlength.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

#CLUSTERING USING MAX FEATURES

#Obtaining bag of words model

cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

#Finding number of max features

l=[]
for i in range(num):
    s=0
    for j in range(1500):
        s=s+X[i][j]
    l.append(s)
stats.dataset['len']=l

#Elbow method

'''
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''

#K-Means for Clustering using max features

kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)
fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 1, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 1, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 1, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')
'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.xlabel('Max features')
plt.ylabel('Sentiment score')
plt.legend()
plt.show()
fig.savefig("../flask/static/images/clustermax.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

#VERIFIED AND NON-VERIFIED USING SENTIMENT SCORES

#Obtaining verified and non-verified accounts count

X=stats.dataset.iloc[:,11].values
vcount=0
ncount=0
for i in range(num):
    if(X[i]==1):
        vcount+=1
    else:
        ncount+=1
        
#Count of positive,negative and neutral tweets 
        
v=[]
n=[]
for i in range(num):
    if(X[i]==1):
        v.append(stats.dataset['compound'][i])
    else:
        n.append(stats.dataset['compound'][i])
vlist=[0,0,0]  
nlist=[0,0,0] 
for i in range(vcount):
    if(v[i]>0):
        vlist[0]+=1
    elif(v[i]==0):
        vlist[1]+=1
    else:
        vlist[2]+=1
for i in range(ncount):
    if(n[i]>0):
        nlist[0]+=1
    elif(n[i]==0):
        nlist[1]+=1
    else:
        nlist[2]+=1
 
#Pie Chart for Verified Tweets
        
labels = 'Positive','Neutral','Negative'
sizes=vlist
explode = (0.1, 0, 0)  
fig1, ax1 = plt.subplots()
colors=['#008080','#CACACA','#575757']
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.show()
fig1.savefig('../flask/static/images/verifiedpie.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#Pie Chart for Non-Verified Tweets
    
labels = 'Positive','Neutral','Negative'
sizes=nlist
explode = (0.1, 0, 0) 
fig2, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  
plt.show()
fig2.savefig('../flask/static/images/nonverifiedpie.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#LANGUAGE DOMINANCE

#Finding number of tweets in each language

Y=stats.dataset.iloc[:,12].values
X=[]
Z=[]
langs=['es','zh','en', 'fr','nl','el','ja','th','hi','ar','pt','tr','tl','fa','et','ta','de','it','in','ru','bn','gu','kn','or','te','sl','ne','ro','lt','mr','pl','ur','ml','ko','ca','pa','vi','da','no','si','sv','cs','fi','ht','iw','eu','bg','cy','hy','am','sr','is','hu','lv','ps','sd','dv','uk','km','lo','ckb','ka']
la=[]
for i in langs:
    c=0
    for j in range(num):
        if(Y[j]==i):
            c=c+1
    X.append(c)

#Creating a dictionary   

dic={}
lan={}
for key in langs:
    for value in X:
        lan[key]=value
        X.remove(value)
        break
Z=[]
dic= sorted(lan.items(), key=lambda x: x[1], reverse=True)
for i in range(62):
    Z.append(dic[i][0])
for j in range(62):
    la.append(dic[j][1])
    
#Finding top five languages count  
    
Z=Z[:5]
la=la[:5]
fig,ax=plt.subplots()
ax.bar(Z[0],la[0],color="#000000")
ax.bar(Z[1],la[1],color="#0000CD")
ax.bar(Z[2],la[2],color="#00FFFF")
ax.bar(Z[3],la[3],color="#ADD8E6")
ax.bar(Z[4],la[4],color="#A9A9A9")     
ax.legend(labels=la) 
plt.xlabel('Languages')
plt.ylabel('Tweets')      
plt.show()
fig.savefig("../flask/static/images/languages.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

#VERIFIED AND NON-VERIFIED

posvalue=[0,0]
negvalue=[0,0]
neuvalue=[0,0]
for i in range(num):
    if(stats.dataset['compound'][i]>0):
        if(stats.dataset['verified'][i]==1):
            posvalue[0]+=1
        else:
            posvalue[1]+=1
    elif(stats.dataset['compound'][i]<0):
        if(stats.dataset['verified'][i]==1):
            negvalue[0]+=1
        else:
            negvalue[1]+=1
    else:
        if(stats.dataset['verified'][i]==1):
            neuvalue[0]+=1
        else:
            neuvalue[1]+=1
            
#Donut for Positive Sentiments
            
labels =['Verified','Non Verified']
sizes=posvalue
colors = ['#1515B4','#3BB9FF']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc="upper right")
plt.title('Positive')
p=plt.gcf()
plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("../flask/static/images/posvernon.png",dpi=100, bbox_inches='tight', pad_inches=0.0)
plt.show()

#Donut for Negative Sentiments

labels =['Verified','Non Verified']
sizes=negvalue
colors = ['#6F4E37','#C85A17']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc="upper right")
plt.title('Negative')
p=plt.gcf()
plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("../flask/static/images/negvernon.png",dpi=100, bbox_inches='tight', pad_inches=0.0)
plt.show()

#Donut for Neutral Sentiments

labels =['Verified','Non Verified']
sizes=neuvalue
colors = ['#EC83D9','#ED1FC4']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts=plt.pie(sizes,colors=colors)
plt.rcParams['text.color'] = 'black'
plt.legend(patches, labels, loc="upper right")
plt.title('Neutral')
p=plt.gcf()
plt.pie(sizes,colors=colors, wedgeprops = { 'linewidth':7, 'edgecolor' : 'white' },autopct='%1.1f%%',pctdistance=0.7)
p.gca().add_artist(my_circle)
p.savefig("../flask/static/images/neuvernon.png",dpi=100, bbox_inches='tight', pad_inches=0.0)
plt.show()

#TOTAL SENTIMENTS

value=[0,0,0]
for i in range(num):
    if(stats.dataset['compound'][i]>0):
        value[0]+=1
    elif(stats.dataset['compound'][i]==0):
        value[1]+=1
    else:
        value[2]+=1
        
#Plotting overall sentiments
        
labels ='Positive','Neutral','Negative'
sizes=value
fig1, ax1 = plt.subplots()
my_circle=plt.Circle( (0,0), 0.5, color='white')
plt.pie(sizes,labels=labels,labeldistance=1.2,autopct='%1.1f%%',pctdistance=0.7,wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
plt.rcParams['text.color'] = 'black'
p=plt.gcf()
p.gca().add_artist(my_circle)
ax.legend(labels=['Positive','Neutral','Negative'])  
p.savefig("../flask/static/images/posneg.png",dpi=100, bbox_inches='tight', pad_inches=0.0)
plt.show()

#IMPACT CLUSTERING

#Elbow Method
'''
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(Y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
'''
#K Means Clustering for followers vs favourite count

Y=stats.dataset.iloc[:,[8,6]].values
kmeans = KMeans(n_clusters =2, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(Y)
fig,ax=plt.subplots()
ax.scatter(Y[y_kmeans == 0, 0],Y[y_kmeans == 0, 1], s = 30, c = 'red', label = 'Cluster 1')
ax.scatter(Y[y_kmeans == 1, 0],Y[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'Cluster 2')
'''
plt.scatter(Y[y_kmeans == 2, 0],Y[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(Y[y_kmeans == 3, 0],Y[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(arr[y_kmeans == 4, 0], arr[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(arr[y_kmeans == 5, 0], arr[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(arr[y_kmeans == 6, 0], arr[y_kmeans == 6, 1], s = 100, c = 'violet', label = 'Cluster 7')
plt.scatter(arr[y_kmeans == 7, 0], arr[y_kmeans == 7, 1], s = 100, c = 'purple', label = 'Cluster 8')
plt.scatter(arr[y_kmeans == 8, 0], arr[y_kmeans == 8, 1], s = 100, c = 'grey', label = 'Cluster 9')
plt.scatter(arr[y_kmeans == 9, 0], arr[y_kmeans == 9, 1], s = 100, c = 'pink', label = 'Cluster 10')
'''
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of people')
plt.ylabel('No. of Followers')
plt.xlabel('Favourites count')
plt.legend()
plt.show()
fig.savefig("../flask/static/images/impactcluster.png",dpi=100, bbox_inches='tight', pad_inches=0.0)


X=stats.dataset.iloc[:,11].values
Y=stats.dataset.iloc[:,16].values

ver=[0,0,0]
nver=[0,0,0]
for i in range(num):
    if(X[i]==1):
        if(Y[i]>0):
            ver[0]=ver[0]+1
        elif(Y[i]<0):
            ver[1]=ver[1]+1
        else:
            ver[2]=ver[2]+1
    else:
        if(Y[i]>0):
            nver[0]=nver[0]+1
        elif(Y[i]<0):
            nver[1]=nver[1]+1
        else:
            nver[2]=nver[2]+1
            
D=[ver,nver]
 
Z=[]
for i in range(len(D[0])): 
        # print(i) 
    row =[] 
    for item in D: 
         row.append(item[i]) 
    Z.append(row) 
    
#graph for verified vs nonverified   
X = np.arange(3)
#fig = plt.figure()
fig,ax=plt.subplots()
#ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, D[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, D[1], color = 'r', width = 0.25)
ax.legend(labels=['Verified', 'Non-Verified'])
plt.xlabel('Sentiment scores')
plt.ylabel('No.of Verified and Non verified users')
#ax.bar(X + 0.50, Z[2], color = 'r', width = 0.25)
plt.show()
fig.savefig("../flask/static/images/pnn.png",dpi=100, bbox_inches='tight', pad_inches=0.0)

#graph for sentiment scores vs verified and nonverified 
X = np.arange(2)
fig = plt.figure()
fig,ax=plt.subplots()
#ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, Z[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, Z[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, Z[2], color = 'r', width = 0.25)
ax.legend(labels=['Positive', 'Negative','Neutral'])
plt.xlabel('Sentiment scores')
plt.ylabel('No.of Verified and Non verified users')
fig.savefig("../flask/static/images/nv.png",dpi=100, bbox_inches='tight', pad_inches=0.0)
