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
#Pour avoir assez d'entrees
newcouple=COORDdata[COORDdata['Volume']>10000].reset_index()
keys=[]
for i in range(len(newcouple)):
    key=(newcouple['location_name'][i],newcouple['Direction'][i])
    keys.append(key)
    
# After trying a first approach on univariate time that didn't seem to give great results, my second approach consists of defining a model over a multivariate time series (multiple couples (location,Direction) since close locations can have influences on each other in traffic volume)

## First Step : Determining Close Locations based on Longitude and Latitude

LISTlonglati=[]
for i in range(len(newcouple)):
    key=[newcouple['location_longitude'][i],newcouple['location_latitude'][i]]
    LISTlonglati.append(key)
LISTlonglati=np.array(LISTlonglati)


#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#kmeans.labels

## location and direction should be strings here
def GetTimeseries(location,direction):
    tsdata=data.loc[data.location_name==location][data.Direction==direction]
    #print(tsdata)
    tsdata=tsdata['Volume']
    #tsdata.plot()
    #plt.show()
    return np.array(tsdata)

def GetGlobalTimeseries(L):
    O=[]
    MAX=[]
    MIN=[]
    MEAN=[]
    for key in L:
        O+=list(GetTimeseries(key[0],key[1])[:11491])
        MAX.append(max(GetTimeseries(key[0],key[1])[:11491]))
        MIN.append(min(GetTimeseries(key[0],key[1])[:11491]))
        MEAN.append(np.mean(GetTimeseries(key[0],key[1])[:11491]))
    MIN=min(MIN)
    MAX=max(MAX)
    MEAN=np.mean(MEAN)
    return np.array(O),MIN,MAX,MEAN
    
#Actually after data crunching, it appears that the max distance between the locations is about 18 km so I decided to feed to the model a multivariate time series.


# PARAMETERS:
# Sliding Step : 2 weeks 
A=24
# Prediction Window: predict 1 week
B=24 #(Maximum 2 jours pour avoir condition relative aux couches B<128*2=256)
# Base training window: (8 weeks here)
m =24*28

# FOR A=1,B=1,m=24*7
# epoch 0 loss in training 0.042862428  loss in test 0.034414649
# epoch 1 loss in training 0.042459846  loss in test 0.034374041
# epoch 2 loss in training 0.042413600  loss in test 0.034949515
# epoch 3 loss in training 0.042598883  loss in test 0.035271741
# epoch 4 loss in training 0.042557107  loss in test 0.035366366


 
#TIMES SERIES avec donnees POUR CHAQUE (Location,Direction,date (donnees chaque heure) 



## First Model CNN (FOR EACH LOCATION AND DIRECTION GET a prediction from the time series)

##Trace des localisations des donnes pour voir s'il est judicieux de trainer le modele sur toutes les localisations (s'ils sont assez proches pour avoir des influences l'une sur l'autre)



class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        #Convolutional Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=38, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        #Convolutional layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=38, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )

## Try to add convolutinal layers ?        
        #Linear Layer 1
        self.fc1 = nn.Linear(in_features=64*8, out_features=64*4)
        #Linear Layer 2
        self.drop=nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(in_features=64*4, out_features=64*3)
        self.fc3 = nn.Linear(in_features=64*3, out_features=24*33)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out=self.drop(out)
        out = self.fc2(out)
        out=self.fc3(out)
        return out





# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#on applique le modele a une seule entree pour voir ce qu'il faut adapter dans un premier lieu

#revoir la normalisation -mean()/max(seq) on all data Module pour normaliser les donner
# Correlation entre Localisations ?
# Prediction de toutes les localisations et directions en meme temps (chaque elements du vecteur correspond a une prediction) 

## (chaque propriete prendre en compte les patterns  )


DIC={}
for i in range(len(keys)):
    DIC[keys[i]]=GetTimeseries(keys[i][0],keys[i][1])




seq=GetGlobalTimeseries(keys)[0]
print(seq.shape)

si2X, si2Y = [], []
#seq here contain only the data
vnorm = GetGlobalTimeseries(keys)[2]-GetGlobalTimeseries(keys)[1]
ME=GetGlobalTimeseries(keys)[3]
dsi2X, dsi2Y = [], []
xlist, ylist = [], []
xx=[]
yy=[]

for k in range(((11491-m-B)//A)-1-int(0.2*(11491//A))):
    xx=[]
    yy=[]
    #maybe-1
    for j in range(33):
        xx += [(seq[j*11491+z]-ME)/vnorm for z in range(k*A,m+k*A)]
        yy += [(seq[j*11491+z]-ME)/vnorm for z in range(m+k*A,m+k*A+B)]
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    ylist.append(torch.tensor(yy,dtype=torch.float32))
si2X = xlist
si2Y= ylist
# Test set

for k1 in range(((11491-m-B)//A)-1-int(0.2*(11491//A)),((11491-B-m)//A)):
    xx=[]
    yy=[]
    #maybe-1
    for j in range(33): # build evaluation dataset 10% 
        xx += [(seq[j*11491+z]-ME)/vnorm for z in range(k1*A,m+k1*A)]
        yy += [(seq[j*11491+z]-ME)/vnorm for z in range(m+k1*A,m+k1*A+B)]
    dsi2Y.append(torch.tensor(yy,dtype=torch.float32))
    dsi2X.append(torch.tensor(xx,dtype=torch.float32))



mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.001)#try 0.0005
xlist = si2X
#if len(xlist)<10:continue
ylist = si2Y
idxtr = list(range(len(xlist)))
for ep in range(2000):
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
    print("epoch %d loss in training %1.9f  loss in test %1.9f" % (ep, lotot, lotestset))
del(mod)
                    

## Train on both of them (B=24)/(B=1) and Pourcentage au lieu de mean squared error take test set rndomly as well


## second CNN model

# FIRST STEP CLUSTERING OF CLOSE ELEMENTS TOGETHER


## Second model LSTM 











