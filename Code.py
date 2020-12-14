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
    return np.array(tsdata,dtype=np.dtype(float))

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
    
#Actually after data crunching, it appears that the max distance between the locations is about 18 km so I decided to feed to the model a multivariate time series.


# PARAMETERS:
# Sliding Step : 2 weeks 
A=1
# Prediction Window: predict 1 week
B=1#(Maximum 2 jours pour avoir condition relative aux couches B<128*2=256)
# Base training window: (8 weeks here)
m =24

#
#For this data A=24*7 i got this
# # Prediction Window: predict 1 week
# B=24*7#(Maximum 2 jours pour avoir condition relative aux couches B<128*2=256)
# # Base training window: (8 weeks here)
# m =24*28*3
# with a learning rate of 0.00009
# epoch 0 loss in training 0.471838177  loss in test 0.077578559
# epoch 1 loss in training 0.423537960  loss in test 0.073999275
# epoch 2 loss in training 0.416005374  loss in test 0.070573498
# epoch 3 loss in training 0.409741406  loss in test 0.075812604
# epoch 4 loss in training 0.408656125  loss in test 0.070852105
# epoch 5 loss in training 0.401123592  loss in test 0.070285265
# epoch 6 loss in training 0.380392734  loss in test 0.082435544
# epoch 7 loss in training 0.355312970  loss in test 0.075832129
# epoch 8 loss in training 0.347168141  loss in test 0.073185963
# epoch 9 loss in training 0.332027352  loss in test 0.075729307
# epoch 10 loss in training 0.324446695  loss in test 0.076210757
# epoch 11 loss in training 0.308708608  loss in test 0.071809540
# epoch 12 loss in training 0.297042384  loss in test 0.076490780
# epoch 13 loss in training 0.291697158  loss in test 0.073425388
# epoch 14 loss in training 0.280970556  loss in test 0.075504074
# epoch 15 loss in training 0.268070364  loss in test 0.076505395
# epoch 16 loss in training 0.257915833  loss in test 0.075552585
# epoch 17 loss in training 0.249529901  loss in test 0.075845813
# epoch 18 loss in training 0.245589789  loss in test 0.076318165
# epoch 19 loss in training 0.228615218  loss in test 0.077533748
# epoch 20 loss in training 0.224274368  loss in test 0.073901700
# epoch 21 loss in training 0.220722528  loss in test 0.072185349
# epoch 22 loss in training 0.210977599  loss in test 0.078165496
# epoch 23 loss in training 0.203596522  loss in test 0.079721950
# epoch 24 loss in training 0.198244498  loss in test 0.084685174
# epoch 25 loss in training 0.197454100  loss in test 0.085641611
# epoch 26 loss in training 0.191584670  loss in test 0.082541567
# epoch 27 loss in training 0.189288178  loss in test 0.081371472
# epoch 28 loss in training 0.185081213  loss in test 0.085814657
# epoch 29 loss in training 0.181005332  loss in test 0.083859200
# epoch 30 loss in training 0.183363169  loss in test 0.078149826
# epoch 31 loss in training 0.178964624  loss in test 0.082515622
# epoch 32 loss in training 0.177221693  loss in test 0.077932429
# epoch 33 loss in training 0.175793831  loss in test 0.079448570
# epoch 34 loss in training 0.171824392  loss in test 0.076258258
# epoch 35 loss in training 0.171378931  loss in test 0.079578316
# epoch 36 loss in training 0.168185166  loss in test 0.077745709
# epoch 37 loss in training 0.167801483  loss in test 0.079506281
# epoch 38 loss in training 0.166070602  loss in test 0.079108103
# epoch 39 loss in training 0.165304903  loss in test 0.089499418
# epoch 40 loss in training 0.162445228  loss in test 0.079746256
# epoch 41 loss in training 0.159858734  loss in test 0.082297324
# epoch 42 loss in training 0.159031569  loss in test 0.075542930
# epoch 43 loss in training 0.160174414  loss in test 0.080739305
# epoch 44 loss in training 0.156031685  loss in test 0.085205454
# epoch 45 loss in training 0.156034311  loss in test 0.085262573
# epoch 46 loss in training 0.155304636  loss in test 0.083840379
# epoch 47 loss in training 0.152952819  loss in test 0.085320485
# epoch 48 loss in training 0.149773690  loss in test 0.084390363
# epoch 49 loss in training 0.146017590  loss in test 0.084415266
# epoch 50 loss in training 0.143693265  loss in test 0.083015878


 
#TIMES SERIES avec donnees POUR CHAQUE (Location,Direction,date (donnees chaque heure) 



## First Model CNN (FOR EACH LOCATION AND DIRECTION GET a prediction from the time series)

##Trace des localisations des donnes pour voir s'il est judicieux de trainer le modele sur toutes les localisations (s'ils sont assez proches pour avoir des influences l'une sur l'autre)



class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        #Convolutional Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        #Convolutional layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )

## Try to add convolutinal layers ?        
        #Linear Layer 1
        self.fc1 = nn.Linear(in_features=64*8, out_features=64*4)
        #Linear Layer 2
        self.drop=nn.Dropout2d(0.0)
        self.fc2 = nn.Linear(in_features=64*4, out_features=2*64)
        self.fc3 = nn.Linear(in_features=2*64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=33)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out=self.drop(out)
        out = self.fc2(out)
        out=self.fc3(out)
        out=self.drop(out)
        out=self.fc4(out)
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

si2X=listtotX[:int(0.9*len(listtotX))]
si2Y=listtotY[:int(0.9*len(listtotY))]

dsi2X =listtotX[int(0.9*len(listtotX)):]
dsi2Y=listtotY[int(0.9*len(listtotY)):]

mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.0005)#try 0.0005
xlist = si2X
#if len(xlist)<10:continue
ylist = si2Y
idxtr = list(range(len(xlist)))
for ep in range(200):
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
del(mod)
                    

## Train on both of them (B=24)/(B=1) and Pourcentage au lieu de mean squared error take test set rndomly as well


## second CNN model

# FIRST STEP CLUSTERING OF CLOSE ELEMENTS TOGETHER


## Second model LSTM 











