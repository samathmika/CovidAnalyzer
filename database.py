import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np
import re
import nltk

#IMPORTING DATASET

dataset=pd.read_csv("corona10.csv",engine="python")
TOT=len(dataset)

def isNaN(string):
    return string != string
X=dataset['tweet']
for i in range(TOT):
    if(isNaN(X[i])):
        X[i]="coronavirus"
dataset['tweet']=X   
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia=SentimentIntensityAnalyzer()
dataset['neg']=dataset['tweet'].apply(lambda x:sia.polarity_scores(x)['neg'])
dataset['neu']=dataset['tweet'].apply(lambda x:sia.polarity_scores(x)['neu'])
dataset['pos']=dataset['tweet'].apply(lambda x:sia.polarity_scores(x)['pos'])
dataset['compound']=dataset['tweet'].apply(lambda x:sia.polarity_scores(x)['compound'])


DATE=dataset.iloc[0,1]
pnn=[0,0,0]
x=dataset['compound']
for i in x:
    if (i >0):
        pnn[0]+=1
    elif(i<0):
        pnn[1]+=1
    else:
        pnn[2]+=1
        
emo=[0,0,0,0] 
y=dataset['labels']       
for i in y:
    if (i=='informative'):
        emo[2]+=1 
    elif (i=='sad'):
        emo[1]+=1 
    elif (i=='angry'):
        emo[0]+=1 
    else:
        emo[3]+=1 


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    create_connection("weekly.db")
    
conn = sqlite3.connect("weekly.db")
print(sqlite3.version)
conn.execute("CREATE TABLE WEEKLY
         (DATE          VARCHAR(20) NOT NULL,
          POS           INT    NOT NULL,
          NEG           INT    NOT NULL,
          NEU           INT    NOT NULL,
          ANGRY         INT      NOT NULL,
          SAD           INT      NOT NULL,
          INFO          INT      NOT NULL,
          HAPPY         INT      NOT NULL,
          TOT           INT      NOT NULL
          );")
print ("Table created successfully")
conn = sqlite3.connect('weekly.db')
conn.execute("INSERT INTO WEEKLY(DATE,POS,NEG,NEU,ANGRY,SAD,INFO,HAPPY,TOT) VALUES(?,?,?,?,?,?,?,?,?)",(DATE,pnn[0],pnn[1],pnn[2],emo[0],emo[1],emo[2],emo[3],TOT))
conn.commit()
conn.close()    
    

conn = sqlite3.connect('weekly.db')

cur = conn.cursor()
var='''SELECT * FROM WEEKLY'''
cur.execute(var)

rows = cur.fetchall()

conn.close()
#conn = sqlite3.connect('weekly.db')
#conn.execute("DELETE FROM WEEKLY WHERE DATE='7/7/2020'")
#conn.commit()

#conn.close()
