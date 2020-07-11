# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 08:31:47 2020

@author: Sowmya
"""

import stats
pnn=[0,0,0]
x=stats.dataset['compound']
for i in x:
    if (i >0):
        pnn[0]+=1
    elif(i<0):
        pnn[1]+=1
    else:
        pnn[2]+=1
        


#Percentage values of sentiments

pos_review=[j for i,j in enumerate(stats.dataset['text']) if stats.dataset['compound'][i]>0.2]
neu_review=[j for i,j in enumerate(stats.dataset['text']) if 0.2>=stats.dataset['compound'][i]>=-0.2]
neg_review=[j for i,j in enumerate(stats.dataset['text']) if stats.dataset['compound'][i]<-0.2]
posp=round(len(pos_review)*100/len(stats.dataset['text']),ndigits=2)
neup=round(len(neu_review)*100/len(stats.dataset['text']),ndigits=2)
negp=round(len(neg_review)*100/len(stats.dataset['text']),ndigits=2)