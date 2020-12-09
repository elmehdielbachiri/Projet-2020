#Author: El Mehdi El Bachiri
#Date 23/11/2020


import torch
import torch.nn as nn
from random import shuffle
import pandas as pd
from random import shuffle

data = pd.read_csv('/Users/mac/Desktop/Projet Machine Learning/Data/Radar_Traffic_Counts.csv',sep=',') 

data.drop(['Time Bin'],axis=1,inplace=True)
data.sort_values(['location_name','Direction','Year','Month','Day','Hour','Minute'],inplace=True)



names=data['location_name'].unique().tolist()
directions=['SB','NB','WB','EB','None']

test = data.loc[data.location_name==names[0]][data.Direction==directions[0]]

#TIMES SERIES avec donnees POUR CHAQUE (Location,Direction,date (donnees chaque heure) 
##location and direction should be strings here
def GetTimeseries(location,direction):
    tsdata=data.loc[data.location_name==location][data.Direction==direction]
    
## First Model CNN

