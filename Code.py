#Author: El Mehdi El Bachiri
#Date 23/11/2020


import torch
import torch.nn as nn
from random import shuffle
import pandas as pd
from random import shuffle

data = pd.read_csv('/Users/mac/Desktop/Projet Machine Learning/Data/Radar_Traffic_Counts.csv',sep=',') 

data.drop(['Time Bin'],axis=1,inplace=True)
data.sort_values(['location_name','Year','Month','Day','Hour','Minute'],inplace=True)

print(data[:1000])


#TIMES SERIES avec donnees POUR CHAQUE (Location,Direction,date (donnees chaque heure) 

## First Model CNN


