# -*- coding: utf-8 -*-
import stats
import datetime
tottweet=len(stats.dataset)

engtweet=0
x=stats.dataset['lang']
for i in x:
    if (i=='en'):
        engtweet+=1
        
'''y=stats.dataset['compound']
pos=0
for i in y:
    if (i>0):
        pos+=1'''
        
info=0 
z=stats.dataset['labels']
for i in z:
    if (i=='informative'):
        info+=1 
        
ver=0
b=stats.dataset['verified']
for i in b:
    if (i==True):
        ver+=1
        
date=datetime.datetime.now().date()
