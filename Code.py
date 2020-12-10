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


# HYPERPARAMETERS:
# Sliding Step :
A=24*7
# Prediction Window:
B=24*7 #(Maximum 10 jours pour avoir condition relative aux couches B<128*2=256)



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
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        #Convolutional layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
        
        #Linear Layer 1
        self.fc1 = nn.Linear(in_features=128*8, out_features=B*4)
        self.drop = nn.Dropout2d(0.25)
        #Linear Layer 2
        self.fc2 = nn.Linear(in_features=B*4, out_features=B*2)
        #Linear Layer 3
        self.fc3 = nn.Linear(in_features=B*2, out_features=B)
 
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
minlen = 24*3*30
# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#on applique le modele a une seule entree pour voir ce qu'il faut adapter dans un premier lieu
si2X, si2Y = [], []
#seq here contain only the data
seq=GetTimeseries(names[0],directions[0])[2]
dsi2X, dsi2Y = [], []
xlist, ylist = [], []
m=24*3*30
print((len(seq)-m)//(24*7)-2)
for k in range((len(seq)-m)//(24*7)-2):
    print(k)
    xx = [seq[z][1]/vnorm for z in range(k*(24*7),m+k*(24*7))]
    if max(xx)>xmax: xmax=max(xx)
    if min(xx)<xmin: xmin=min(xx)
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    yy = [seq[z][1]/vnorm for z in range(m+k*(24*7),m+(k+1)*(24*7))]
    ylist.append(torch.tensor(yy,dtype=torch.float32))
si2X = xlist
si2Y= ylist
if True: # build evaluation dataset
    k1=(len(seq)-m)//(24*7)-2
    xx = [seq[z][1]/vnorm for z in range(k1*(24*7),m+k1*(24*7))]
    dsi2X = [torch.tensor(xx,dtype=torch.float32)]
    yy = [seq[z][1]/vnorm for z in range(m+k1*(24*7),m+(k1+1)*(24*7))]
    dsi2Y = [torch.tensor(yy,dtype=torch.float32)]




mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.01)
xlist = si2X
#if len(xlist)<10:continue
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
    haty = mod(dsi2X[0].view(1,1,-1))
    lo = loss(haty,dsi2Y[0].view(1,-1))
    print("epoch %d loss %1.9f testMSE %1.9f" % (ep, lotot, lo.item()))




## Second model LSTM 











