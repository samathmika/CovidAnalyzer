import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import database


X=np.array(rows)
date=X[:,0]
pos=X[:,1]
pos=pos.tolist()
neg=X[:,2]
neg=neg.tolist()
neu=X[:,3]
neu=neu.tolist()
angry=X[:,4]
angry=angry.tolist()
sad=X[:,5]
sad=sad.tolist()
info=X[:,6]
info=info.tolist()
happy=X[:,7]
happy=happy.tolist()
tot=X[:,8]
tot=tot.tolist()

for i in range(len(pos)):
    pos[i]=int(pos[i])
    neg[i]=int(neg[i])
    neu[i]=int(neu[i])
    angry[i]=int(angry[i])
    sad[i]=int(sad[i])
    info[i]=int(info[i])
    happy[i]=int(happy[i])
    tot[i]=int(tot[i])


#sad tweets

fig2,ax=plt.subplots()
plt.bar(date,sad,color="#fab346")
plt.xlabel('Date')
plt.ylabel('No of Sad Tweets')
plt.show()
fig2.savefig('../flask/static/images/datesad1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#angry tweets
fig2,ax=plt.subplots()
plt.bar(date,angry,color="red")
plt.xlabel('Date')
plt.ylabel('No of Angry Tweets')
plt.show()
fig2.savefig('../flask/static/images/dateangry1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)

#info tweets
fig2,ax=plt.subplots()
plt.bar(date,info,color="#88dcfe")
plt.xlabel('Date')
plt.ylabel('No of Informative Tweets')
plt.show()
fig2.savefig('../flask/static/images/dateinfo1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


#Pos,Neg and Neu 

fig2,ax=plt.subplots()
plt.plot(date,pos,color="blue",label='Positive')
plt.plot(date,neg,color="black",label='Negative')
plt.plot(date,neu,color="red",label='Neutral')
plt.xlabel('Date')
plt.ylabel('No of Tweets')
plt.legend()
plt.show()
fig2.savefig('../flask/static/images/datepnn1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


#Labels

fig2,ax=plt.subplots()
plt.plot(date,angry,color="blue",label='Angry')
plt.plot(date,happy,color="black",label='Happy')
plt.plot(date,sad,color="red",label='Sad')
plt.plot(date,info,color="green",label='Informative')
plt.xlabel('Date')
plt.ylabel('No of Tweets')
plt.legend()
plt.show()
fig2.savefig('../flask/static/images/datelabel1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


#Positive

fig2,ax=plt.subplots()
plt.bar(date,pos,color="Purple")
plt.xlabel('Date')
plt.ylabel('No of Positive Tweets')
plt.show()
fig2.savefig('../flask/static/images/datepos1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


#Happy

fig2,ax=plt.subplots()
plt.bar(date,happy,color="#fc55d5")
plt.xlabel('Date')
plt.ylabel('No of Happy Tweets')
plt.show()
fig2.savefig('../flask/static/images/datehappy1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


#Total Tweets

fig2,ax=plt.subplots()
plt.bar(date,tot,color="Cyan")
plt.xlabel('Date')
plt.ylabel('Total Tweets')
plt.show()
fig2.savefig('../flask/static/images/datetot1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)


#7

fig2,ax=plt.subplots()
ax1 = plt.subplot(111)
plt.plot(date,angry,color="blue",label='Angry')
plt.plot(date,happy,color="black",label='Happy')
plt.plot(date,sad,color="red",label='Sad')
plt.plot(date,info,color="green",label='Informative')
plt.plot(date,pos,color="cyan",label='Positive')
plt.plot(date,neg,color="purple",label='Negative')
plt.plot(date,neu,color="#fc55d5",label='Neutral')
plt.xlabel('Date')
plt.ylabel('No of Tweets')
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1.5
legend_y = 0.5
plt.legend(loc="center right",bbox_to_anchor=(legend_x, legend_y))
plt.show()
fig2.savefig('../flask/static/images/dateall1.png',dpi=100, bbox_inches='tight', pad_inches=0.0)












            
