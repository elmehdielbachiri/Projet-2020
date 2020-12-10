#Author: El Mehdi El Bachiri
#Date 23/11/2020

import numpy as np
import torch
import torch.nn as nn
from random import shuffle
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/mac/Desktop/Projet Machine Learning/Data/Radar_Traffic_Counts.csv',sep=',') 

data.drop(['Time Bin'],axis=1,inplace=True)
date1=data[['Year','Month','Day','Hour']]
data.drop('Month',axis=1,inplace=True)
data.drop('Day',axis=1,inplace=True)
data.drop('Hour',axis=1,inplace=True)
data.drop('Minute',axis=1,inplace=True)
data.drop('Day of Week',axis=1,inplace=True)
data.drop('location_longitude',axis=1,inplace=True)
data.drop('location_latitude',axis=1,inplace=True)


data[['Year']]=pd.to_datetime(date1,unit='D')
data=data.rename(columns={'Year':'Date'})
data=data.groupby(by=['location_name','Direction','Date'],as_index=False)['Volume'].sum()
data.sort_values(['location_name','Direction','Date'],inplace=True)
data.head()

names=data['location_name'].unique().tolist()
directions=data['Direction'].unique().tolist()

test = data.loc[data.location_name==names[0]][data.Direction==directions[0]]

#TIMES SERIES avec donnees POUR CHAQUE (Location,Direction,date (donnees chaque heure) 
##location and direction should be strings here
def GetTimeseries(location,direction):
    tsdata=data.loc[data.location_name==location][data.Direction==direction]
    tslist=[]
    for i in range(len(tsdata)):
        tslist+=[(tsdata['Date'][i],tsdata['Volume'][i])]
    tsdata=pd.Series(np.array(tsdata['Volume']),index=tsdata['Date'])
    #tsdata.plot()
    #plt.show()
    return tsdata,np.array(tsdata),tslist


## First Model CNN

class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        #Convolutional Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        #Convolutional layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
        
        #Linear Layer 1
        self.fc1 = nn.Linear(in_features=64*8, out_features=24*7*4*2)
        self.drop = nn.Dropout2d(0.25)
        #Linear Layer 2
        self.fc2 = nn.Linear(in_features=24*7*4*2, out_features=24*7*4)
        #Linear Layer 3
        self.fc3 = nn.Linear(in_features=24*7*4, out_features=24*7)
 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



xmin, xmax = 100.0, -100.0
vnorm = 1000.0
minlen = 15000
# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#on applique le modele a une seule entree pour voir ce qu'il faut adapter dans un premier lieu
si2X, si2Y = [], []
#seq here contain only the data
seq=GetTimeseries(names[0],directions[0])[2]
dsi2X, dsi2Y = [], []
xlist, ylist = [], []
for m in range(minlen, len(seq)-1):
    print(m)
    xx = [seq[z][1]/vnorm for z in range(m)]
    if max(xx)>xmax: xmax=max(xx)
    if min(xx)<xmin: xmin=min(xx)
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    yy = [seq[m+1][1]/vnorm]
    ylist.append(torch.tensor(yy,dtype=torch.float32))
    si2X = xlist
    si2Y= ylist
    if True: # build evaluation dataset
        xx = [seq[z][1]/vnorm for z in range(len(seq)-1)]
        dsi2X = [torch.tensor(xx,dtype=torch.float32)]
        yy = [seq[len(seq)-1][1]/vnorm]
        dsi2Y = [torch.tensor(yy,dtype=torch.float32)]




mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.01)
xlist = si2X
#if len(xlist)<10:continue
ylist = si2Y
idxtr = list(range(len(xlist)))
for ep in range(5):
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
    haty = mod(dsi2X[0].view(1,1,-1))
    lo = loss(haty,dsi2Y[0].view(1,-1))
    print("epoch %d loss %1.9f testMSE %1.9f" % (ep, lotot, lo.item()))




## Second model LSTM 











