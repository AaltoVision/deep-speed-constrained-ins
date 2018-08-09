# dataset definition
#
# Description:
#   Define dataset and import data.
#
# Copyright (C) 2018 Santiago Cortes
#
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import csv

# Dataset class
class OdometryDataset(Dataset):
    def __init__(self, data_folder,datasets, transform=None):
        """
        Args:
            data_folder (string): Path to the csv file with annotations.
            datasets: list of datasets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ind=0
        self.imu=[]
        self.imut=[]
        self.pos=[]
        self.post=[]
        self.limits=[]
        self.limits.append(0)
        plot=False
        #scroll trough folders and attach data. Since there is not that many sequences, one array is used.
        for dataset in datasets:
            imu_path=data_folder+dataset+"iphone/imu-gyro.csv"
            data = pd.read_csv(imu_path,names=list('tlabcdefghijk'))
            pos=data[data['l']==7]
            imu=data[data['l']==34]      
            self.imut.append(imu[list('t')])
            self.imu.append(imu[list('abcdef')])
            self.post.append(pos[list('t')])
            self.pos.append(pos[list('bcd')])    
            self.transform = transform
            self.limits.append(self.limits[ind]+len(self.imu[ind])-300)
      
            if plot:
                plt.plot(self.pos[ind].values)
                print(np.shape(np.diff(self.pos[ind].values,axis=0,n=1)))
                plt.figure()
                dt=np.diff(self.post[ind].values,axis=0)
                print(np.shape(dt))
                plt.plot(np.mean((np.diff(self.pos[ind],axis=0)/dt[:,None]),0))
                plt.figure()
                plt.plot(self.imu[ind].values)
                plt.show()                
            ind=ind+1
       

    # Define the length of the dataset as the number of sequences that can be extracted.  
    def __len__(self):

        return np.floor((self.limits[-1]-1)/100)
    
    # read a sample of 100 measurements, avoiding the edges of the diferent captures.
    def __getitem__(self, idx):
        if idx>len(self):
            raise ValueError('Index out of range')
        else:
            idx=idx*100
                
        for index in range(0,len(self.limits)):
            if idx>=self.limits[index] and idx<self.limits[index+1]:

                dset=index
                off=np.random.randint(low=50,high=100)
                idx=idx-self.limits[index]+off
                break
        
        IMU=self.imu[dset][idx:idx+200].values
        acc=IMU[0:3][1]

        IMU=IMU.swapaxes(0, 1)

        t=(self.imut[dset])[idx:idx+200].values

        ti=np.min(t)
        te=np.max(t)

        inde=np.logical_and([self.post[dset]['t'].values<te] , [self.post[dset]['t'].values>ti])
        inde=np.squeeze(inde)

        
        posi=self.pos[dset][inde].values
        dt=np.diff(self.post[dset][inde].values,axis=0)
        dp=np.diff(posi,axis=0)
        T=self.post[dset][inde].values
        dT=T[-1]-T[0]
        dP=posi[:][-1]-posi[:][0]

        minv=np.min(np.sqrt(np.sum(np.square(dp/dt),axis=1)))
        maxv=np.max(np.sqrt(np.sum(np.square(dp/dt),axis=1)))
        
        gt=np.mean((dp/dt),axis=0)
        gt=dP/dT
        

        gt=gt.astype('float')
        IMU=IMU.astype('float')
        sample={'imu':IMU,'gt':gt,'time':T[0],'range':[minv,maxv]}
        if self.transform:
            sample = self.transform(sample)
            
        return sample

        # Trasform sample into tensor structure.
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        imu=sample['imu']
        gt=sample['gt']
        T=sample['time']
        R=sample['range']
        #print(type(R))
        return {'imu': torch.from_numpy(imu),'gt':torch.from_numpy(gt),'time':T,'range':R}
