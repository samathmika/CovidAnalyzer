#IMPORTING LIBRARIES

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#IMPORTING DATASET

data=pd.read_csv("tweets.csv",engine="python")
num=len(data)

#EXTRACTING HOURS,MINUTES AND SECONDS

h,m,s=[],[],[]
X=data.iloc[:,4].values
X=X.tolist()
for i in range(num):
    a,b,c=X[i].split(':')
    h.append(a)
    m.append(b)
    s.append(c)

#OBTAINING TWEET COUNT
    
count=[]
for i in range(48):
    count.append(0)
for i in range(len(h)):
    if(h[i]=='0'):
        h[i]='24'
for i in range(24,9,-1):
    for j in h:
        if(j==str(i)):
            count[i]+=1
            
#PLOTTING GRAPH FOR TWEET COUNT VS TIME
            
count=count[10:25]
label=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]            
plt.figure(figsize=(10,6))
fig,ax = plt.subplots()
ax.set_facecolor("#0c021a")
x=np.array(label)
y=np.array(count)
x_new = np.linspace(x.min(), x.max(),500)
f = interp1d(x, y, kind='quadratic')
y_smooth=f(x_new)
ax.scatter (x, y,marker='*',color='violet',s=100)
ax.plot (x_new,y_smooth,color="purple",linewidth=3)
plt.xlabel("Time(in hours)")
plt.ylabel("No. of Tweets")
plt.show()
fig.savefig("../flask/static/images/daywise.png",dpi=100, bbox_inches='tight', pad_inches=0.0)











