import pandas as pd

dataset=pd.read_csv("corona1.csv",engine="python")
favorites=dataset.iloc[:,6].values
followers=dataset.iloc[:,8].values


#Maximum favourite count
maxi=0
for j in range(463418):
    if(favorites[j]>maxi):
        maxi=favorites[j]
        f=j
favb=dataset['screen_name'][f]

#Max fav min followers

followers=followers.tolist()
s=max(followers)
m=0
for j in range(463418):
    if(followers[j]>maxi):
        maxi=favorites[j]
        f=j
ma=0
na=s
for i in range(463418):
    if(favorites[i]>ma):
        if(followers[i]<na):
            ma=favorites[i]
            na=followers[i]
            d=i
favbfols=dataset['screen_name'][d]    

#Min fav max followers

favorites=favorites.tolist()
e=max(favorites)
u=0
v=e
for i in range(463418):
    if(followers[i]>u):
        if(favorites[i]<v):
            u=followers[i]
            v=favorites[i]
            q=i

folbfavs=dataset['screen_name'][q]   