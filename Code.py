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


# PARAMETERS:
# Sliding Step : 2 weeks
A=24*14
# Prediction Window: predict 1 week
B=24 #(Maximum 2 jours pour avoir condition relative aux couches B<128*2=256)
# Base training window: (8 weeks here)
m = 24*28

# FOR : A=24*7
# Prediction Window: predict 1 week
# B=24 #(Maximum 2 jours pour avoir condition relative aux couches B<128*2=256)
# Base training window: (8 weeks here)
# m = 24*28*2
# epoch 0 loss 13.663952358 testMSE 0.036261350
# epoch 1 loss 11.580940861 testMSE 0.106750794
# epoch 2 loss 11.368167730 testMSE 0.053207424
# epoch 3 loss 12.217421100 testMSE 0.059360106
# epoch 4 loss 11.638190819 testMSE 0.018149979
# epoch 5 loss 11.520276275 testMSE 0.058282465
# epoch 6 loss 11.319544181 testMSE 0.039052293
# epoch 7 loss 11.287821383 testMSE 0.053247068
# epoch 8 loss 11.165210446 testMSE 0.075795271
# epoch 9 loss 11.256984541 testMSE 0.036187522
# epoch 10 loss 11.282688415 testMSE 0.051304504
# epoch 11 loss 11.216889381 testMSE 0.056534965
# epoch 12 loss 11.033790175 testMSE 0.048409667
# epoch 13 loss 11.458225170 testMSE 0.064952679
# epoch 14 loss 11.130934507 testMSE 0.031871408
# epoch 15 loss 12.486707686 testMSE 0.042964790
# epoch 16 loss 12.092693646 testMSE 0.036532227
# epoch 17 loss 11.621955955 testMSE 0.063966967
# epoch 18 loss 11.553644409 testMSE 0.046707377
# epoch 19 loss 11.023357280 testMSE 0.061495472 


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
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
        
        #Linear Layer 1
        self.fc1 = nn.Linear(in_features=128*8, out_features=B*2)
        self.drop = nn.Dropout2d(0.25)
        #Linear Layer 2
        self.fc2 = nn.Linear(in_features=B*2, out_features=B*2)
        #Linear Layer 3 (USEFUL OR NOT ? 2 linear layers are useful)
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

# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#on applique le modele a une seule entree pour voir ce qu'il faut adapter dans un premier lieu
si2X, si2Y = [], []
#seq here contain only the data
seq=GetTimeseries(names[0],directions[0])[2]
dsi2X, dsi2Y = [], []
xlist, ylist = [], []
print((len(seq)-m//A)-2)
for k in range(((len(seq)-m-B)//A)-1):
    print(k)
    xx = [seq[z][1]/vnorm for z in range(k*A,m+k*A)]
    if max(xx)>xmax: xmax=max(xx)
    if min(xx)<xmin: xmin=min(xx)
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    yy = [seq[z][1]/vnorm for z in range(m+k*A,m+k*A+B)]
    ylist.append(torch.tensor(yy,dtype=torch.float32))
si2X = xlist
si2Y= ylist
# Test set
if True: # build evaluation dataset 10% 
    k1=((len(seq)-m-B)//A)-1
    xx = [seq[z][1]/vnorm for z in range(k1*A,m+k1*A)]
    dsi2X = [torch.tensor(xx,dtype=torch.float32)]
    yy = [seq[z][1]/vnorm for z in range(m+k1*A,m+k1*A+B)]
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


## TEST both of them (B=24)/ (B=1)

## Second model LSTM 











