#Author: El Mehdi El Bachiri
#Date 23/11/2020

import numpy as np
import torch
import torch.nn as nn
from random import shuffle
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('/Users/mac/Desktop/Projet Machine Learning/Data/Radar_Traffic_Counts.csv',sep=',') 


### First Step Data Preprocessing (we transform data into hourly time series)

data.drop(['Time Bin'],axis=1,inplace=True)
date1=data[['Year','Month','Day','Hour']]
data.drop('Month',axis=1,inplace=True)
data.drop('Day',axis=1,inplace=True)
data.drop('Hour',axis=1,inplace=True)
data.drop('Minute',axis=1,inplace=True)
data.drop('Day of Week',axis=1,inplace=True)


#data.drop('location_longitude',axis=1,inplace=True)
#data.drop('location_latitude',axis=1,inplace=True)

data[['Year']]=pd.to_datetime(date1,unit='D')
data=data.rename(columns={'Year':'Date'})
data=data.groupby(by=['location_name','Direction','location_latitude','location_longitude','Date'],as_index=False)['Volume'].sum()
data.sort_values(['location_name','Direction','Date'],inplace=True)
data.head()

names=data['location_name'].unique().tolist()
directions=data['Direction'].unique().tolist()

test = data.loc[data.location_name==names[0]][data.Direction==directions[0]]

COORDdata=data.groupby(by=['location_name','location_latitude','location_longitude','Direction'],as_index=False)['Volume'].count()
# To have enough entries we only keep the couples location, direction that verify number of hourly volume values >10 000
newcouple=COORDdata[COORDdata['Volume']>10000].reset_index()
keys=[]
for i in range(len(newcouple)):
    key=(newcouple['location_name'][i],newcouple['Direction'][i])
    keys.append(key)
    
# After trying a first approach on univariate time series (for each couple location,direction find a prediction) that didn't seem to give great results, my second approach consists of defining a model over a multivariate time series (multiple couples (location,Direction) since close locations can have influences on each other in traffic volume)

## First Step : Determining Close Locations based on Longitude and Latitude

LISTlonglati=[]
for i in range(len(newcouple)):
    key=[newcouple['location_longitude'][i],newcouple['location_latitude'][i]]
    LISTlonglati.append(key)
LISTlonglati=np.array(LISTlonglati)


#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#kmeans.labels

# After a step of distance computation it seemed that the max distance between the locations is 17km so this approach is quite relevant

## location and direction should be strings here
def GetTimeseries(location,direction):
    tsdata=data.loc[data.location_name==location][data.Direction==direction]
    #print(tsdata)
    tsdata=tsdata['Volume']
    #tsdata.plot()
    #plt.show()
    return np.array(tsdata,dtype=np.dtype(float))


# here we define a global "time series" by horizantally stacking all the times series over 11491 which represents the min of length of the stacked time series 

def GetGlobalTimeseries(L):
    O=[]
    for key in L:
        K=GetTimeseries(key[0],key[1])[:11491]
        MAX=max(K)
        MIN=min(K)
        MEAN=np.mean(K)
        for i in range(len(K)):
            a=(K[i]-MEAN)/(MAX-MIN)
            K[i]=a
        O+=list(K)
    return np.array(O)
    



# PARAMETERS: THESE WILL BE VERY IMPORTANT TO DETERMINE RANGES THAT WE GO THROUGH IN THE DATA
# Sliding Step : 1 hour 
A=1
# Prediction Window: predict next hour (for all the locations 33 outputs in the neural network)
B=1#
# Base training window: (24 hours here)
m =24
# 
# #
# epoch 0 loss in training 221.333390981 mean of loss in training 0.024131421 loss in test 64.110077986 mean loss in test 0.027946852
# epoch 1 loss in training 148.670976395 mean of loss in training 0.016209221 loss in test 57.730604505 mean loss in test 0.025165913
# epoch 2 loss in training 133.384240406 mean of loss in training 0.014542547 loss in test 52.436853564 mean loss in test 0.022858262
# epoch 3 loss in training 124.386483121 mean of loss in training 0.013561544 loss in test 51.762438581 mean loss in test 0.022564271
# epoch 4 loss in training 118.565050375 mean of loss in training 0.012926848 loss in test 52.853466719 mean loss in test 0.023039872
# epoch 5 loss in training 114.318383842 mean of loss in training 0.012463845 loss in test 49.730500669 mean loss in test 0.021678509
# epoch 6 loss in training 110.582567063 mean of loss in training 0.012056538 loss in test 48.958620852 mean loss in test 0.021342032
# epoch 7 loss in training 107.992608879 mean of loss in training 0.011774161 loss in test 48.691610710 mean loss in test 0.021225637
# epoch 8 loss in training 105.835278420 mean of loss in training 0.011538953 loss in test 48.052927462 mean loss in test 0.020947222
# epoch 9 loss in training 103.927426717 mean of loss in training 0.011330945 loss in test 46.784571065 mean loss in test 0.020394320
# epoch 10 loss in training 102.135780208 mean of loss in training 0.011135606 loss in test 46.514949507 mean loss in test 0.020276787
# epoch 11 loss in training 100.229318773 mean of loss in training 0.010927750 loss in test 47.197454480 mean loss in test 0.020574304
# epoch 12 loss in training 98.756169510 mean of loss in training 0.010767136 loss in test 45.639828970 mean loss in test 0.019895305
# epoch 13 loss in training 97.486862930 mean of loss in training 0.010628747 loss in test 46.520865181 mean loss in test 0.020279366
# epoch 14 loss in training 96.303059712 mean of loss in training 0.010499679 loss in test 45.089780811 mean loss in test 0.019655528
# epoch 15 loss in training 95.154366519 mean of loss in training 0.010374440 loss in test 43.728359205 mean loss in test 0.019062057
# epoch 16 loss in training 93.863133689 mean of loss in training 0.010233660 loss in test 46.855021201 mean loss in test 0.020425031
# epoch 17 loss in training 93.022386499 mean of loss in training 0.010141996 loss in test 44.105041426 mean loss in test 0.019226260
# epoch 18 loss in training 91.745043340 mean of loss in training 0.010002730 loss in test 47.039255780 mean loss in test 0.020505343
# epoch 19 loss in training 90.849961996 mean of loss in training 0.009905142 loss in test 44.231947942 mean loss in test 0.019281581
# epoch 20 loss in training 90.040892264 mean of loss in training 0.009816931 loss in test 44.198303514 mean loss in test 0.019266915
# epoch 21 loss in training 89.312546569 mean of loss in training 0.009737521 loss in test 44.467598668 mean loss in test 0.019384306
# epoch 22 loss in training 88.393852283 mean of loss in training 0.009637359 loss in test 43.653075736 mean loss in test 0.019029240
# epoch 23 loss in training 87.451857317 mean of loss in training 0.009534655 loss in test 43.631733544 mean loss in test 0.019019936
# epoch 24 loss in training 86.845682766 mean of loss in training 0.009468566 loss in test 42.332912820 mean loss in test 0.018453754
# epoch 25 loss in training 86.217479718 mean of loss in training 0.009400074 loss in test 43.261572119 mean loss in test 0.018858575
# epoch 26 loss in training 85.773683547 mean of loss in training 0.009351688 loss in test 42.595937441 mean loss in test 0.018568412
# epoch 27 loss in training 85.109763443 mean of loss in training 0.009279303 loss in test 42.499135893 mean loss in test 0.018526214
# epoch 28 loss in training 84.481761430 mean of loss in training 0.009210833 loss in test 42.462293635 mean loss in test 0.018510154
# epoch 29 loss in training 83.916044442 mean of loss in training 0.009149154 loss in test 41.087451736 mean loss in test 0.017910833
# epoch 30 loss in training 83.556549465 mean of loss in training 0.009109960 loss in test 41.692256741 mean loss in test 0.018174480

 



## I used a CNN model


class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        #Convolutional Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        #Convolutional layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
## Try to add convolutinal layers ?        
        #Linear Layer 1
        self.fc1 = nn.Linear(in_features=256*8, out_features=128*8)
        #Linear Layer 2
        self.drop=nn.Dropout2d(0.0)
        self.fc2 = nn.Linear(in_features=128*8, out_features=128*4)
        self.fc3 = nn.Linear(in_features=128*4, out_features=33)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out=self.drop(out)
        out = self.fc2(out)
        out=self.fc3(out)
        return out



## The dictionary here was used for my first approach of univariate time series

DIC={}
for i in range(len(keys)):
    DIC[keys[i]]=GetTimeseries(keys[i][0],keys[i][1])


## Here we apply the model over the defined stack of time series

seq=GetGlobalTimeseries(keys)
print(seq.shape)

si2X, si2Y = [], []
#seq here contain only the data
xlist, ylist = [], []


for k in range((11491-B-m)//A):
    xx=[]
    yy=[]
    for j in range(33):
        xx += [seq[j*11491+z] for z in range(k*A,m+k*A)]
        yy += [seq[j*11491+z] for z in range(m+k*A,m+k*A+B)]
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    ylist.append(torch.tensor(yy,dtype=torch.float32))
listtotX= xlist
listtotY= ylist
# Test set
#shuffle(listtotX)
#shuffle(listtotY)
##

#TRAINING SET
si2X=listtotX[:int(0.8*len(listtotX))]
si2Y=listtotY[:int(0.8*len(listtotY))]

#TEST SET (roughly 20%) 
dsi2X =listtotX[int(0.8*len(listtotX)):]
dsi2Y=listtotY[int(0.8*len(listtotY)):]


Y=[]
X=[]
E=[]
mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.0001)
xlist = si2X
ylist = si2Y
idxtr = list(range(len(xlist)))
for ep in range(20):
    shuffle(idxtr)
    lotot=0.
    mod.train()
    for j in idxtr:
        opt.zero_grad()
        haty = mod(xlist[j].view(1,1,-1))
        # print("pred %f" % (haty.item()*vnorm))
        lo = loss(haty,ylist[j].view(1,-1))
        lotot += lo.item()
        lo.backward()
        opt.step()
        
# the MSE here is computed on a single sample: so it's highly variable !
        # to make sense of it, you should average it over at least 1000 (s,i) points
    mod.eval()
    lotestset=0
    for i in range(len(dsi2X)):
        haty = mod(dsi2X[i].view(1,1,-1))
        lo = loss(haty,dsi2Y[i].view(1,-1))
        lotestset+= lo.item()
    print("epoch %d loss in training %1.9f mean of loss in training %1.9f loss in test %1.9f mean loss in test %1.9f" % (ep, lotot,lotot/len(xlist), lotestset,lotestset/len(dsi2X)))
    E.append(ep)
    X.append(lotot/len(xlist))
    Y.append(lotestset/len(dsi2X))

plt.plot(E,X,'r--',E,Y,'bs')
plt.show()
del(mod)
                    












