# Import used libraries.
import torch
import pandas as pd
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import subprocess
import time
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
        for dataset in datasets:
            capture=data_folder+"capture"+dataset+"iphone/*.csv"
            imu_path=subprocess.check_output('ls '+capture,shell=True)
        
            imu_path=data_folder+"capture"+dataset+"iphone/data_for_rosbag.csv"
            data = pd.read_csv(imu_path,names=list('tlabcdefghijk'))
            pos=data[data['l']==7]
            imu=data[data['l']==34]
        
            self.imut.append(imu[list('t')])
            self.imu.append(imu[list('abcdef')])
            #self.imu=self.imu.loc[self.imu['0'].isin(['34'])]
            self.post.append(pos[list('t')])
            self.pos.append(pos[list('bcd')])    
        
            #gt_path=data_folder+"results"+dataset+"gt.csv"
            #self.gt = pd.read_csv(gt_path)
            self.transform = transform
            self.limits.append(self.limits[ind]+len(self.imu[ind])-300)
            
            
            
            if plot:
                plt.plot(self.pos[ind].values)
                print(np.shape(np.diff(self.pos[ind].values,axis=0,n=1)))
                plt.figure()
                #plt.plot(np.diff(self.pos.values,axis=0,n=1))
                dt=np.diff(self.post[ind].values,axis=0)
                print(np.shape(dt))
                plt.plot(np.mean((np.diff(self.pos[ind],axis=0)/dt[:,None]),0))
                plt.figure()
                plt.plot(self.imu[ind].values)
                plt.show()
                
            ind=ind+1
       

        

    def __len__(self):
        #print(self.limits)
        return np.floor((self.limits[-1]-1)/100)
    

    def __getitem__(self, idx):
        if idx>len(self):
            raise ValueError('Index out of range')
        else:
            idx=idx*100
                
        for index in range(0,len(self.limits)):
            if idx>=self.limits[index] and idx<self.limits[index+1]:
                #print(idx)
                #print(self.limits[index])
                dset=index
                off=np.random.randint(low=5,high=100)
                idx=idx-self.limits[index]+off
                break
        
        IMU=self.imu[dset][idx:idx+200].values
        acc=IMU[0:3][1]
        #IMU=np.expand_dims(IMU,2).astype('float')
        IMU=IMU.swapaxes(0, 1)
        #print(np.shape(IMU))
        #print(self.imu[idx:idx+200])
        t=(self.imut[dset])[idx:idx+200].values
        #print(idx)
        ti=np.min(t)
        te=np.max(t)
        #print(ti)
        #print(te)
        inde=np.logical_and([self.post[dset]['t'].values<te] , [self.post[dset]['t'].values>ti])
        inde=np.squeeze(inde)
        #plt.plot(inde )
        #print(type(self.post['t'].values<te))
        
        posi=self.pos[dset][inde].values
        dt=np.diff(self.post[dset][inde].values,axis=0)
        dp=np.diff(posi,axis=0)
        T=self.post[dset][inde].values
        dT=T[-1]-T[0]
        dP=posi[:][-1]-posi[:][0]
        #print(dP)
        #print(np.shape(posi))
        #print(np.shape(np.diff(posi,0)))
        #print(np.mean(np.diff(posi,0),0))
        #print(np.shape((IMU)))
        #plt.figure()
        #plt.plot(np.squeeze(IMU))
        #plt.show()
        #print(np.shape(gt))
        #print(np.mean(np.diff(gt,0),0))
        minv=np.min(np.sqrt(np.sum(np.square(dp/dt),axis=1)))
        maxv=np.max(np.sqrt(np.sum(np.square(dp/dt),axis=1)))
        
        gt=np.mean((dp/dt),axis=0)
        gt=dP/dT
        
        #print(minv)
        #print(np.sqrt(np.sum(np.square(gt))))
        #print(maxv)
        
        
        #print(np.shape((dp/dt)))
        gt=gt.astype('float')
        IMU=IMU.astype('float')
        sample={'imu':IMU,'gt':gt,'time':T[0],'range':[minv,maxv]}
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        imu=sample['imu']
        gt=sample['gt']
        T=sample['time']
        R=sample['range']
        #print(type(R))
        return {'imu': torch.from_numpy(imu),'gt':torch.from_numpy(gt),'time':T,'range':R}
