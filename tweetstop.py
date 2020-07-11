# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:34:30 2020

@author: Sowmya
"""

import stats
x=stats.dataset['labels']

info=[0,0,0]
sad=[0,0,0]
angry=[0,0,0]
happy=[0,0,0] 
for i in x:
    if (i=='informative'):
        info[0]+=1 
    elif (i=='sad'):
        sad[0]+=1 
    elif (i=='angry'):
        angry[0]+=1 
    else:
        happy[0]+=1 

y=stats.dataset['verified']
j=0
for i in x:
    if (i=='informative'):
        if(y[j]==1):
            info[1]+=1 
        else:
            info[2]+=1
    elif (i=='sad'):
        if(y[j]==1):
            sad[1]+=1 
        else:
            sad[2]+=1
    elif (i=='happy'):
        if(y[j]==1):
            happy[1]+=1 
        else:
            happy[2]+=1
    elif (i=='angry'):
        if(y[j]==1):
            angry[1]+=1 
        else:
            angry[2]+=1
    j+=1

