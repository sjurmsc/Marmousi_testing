# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:25:06 2020

@author: JRD
"""

# sys.path.append('C:\\Users\\JRD\\OneDrive - NGI\\Python project\\Autre')

import numpy as np
import numpy.linalg.linalg as lin
import matplotlib.pyplot as plt
import math, sys, time, random
from scipy.interpolate import interp1d
from scipy import fft  
import scipy.io as sio


##############################################
#### Translation of reflectivity to ai and to JP code

def reflectivity_to_ai(refl,slope=None,basevalue=1500):
    '''
    Reflectivity to Acoustic Impedance
    Can take a slope profile into account for linear trend in the layers
    '''
    if slope is None:
        slope=np.zeros(len(refl))
        
    ai=np.ones(np.shape(refl))*basevalue
    for i in range(len(refl)-1):
        ai[i+1]=(1+refl[i+1])/(1-refl[i+1])*ai[i]
        ai[i+1]=ai[i+1]+slope[i+1]
        
    return ai

from math import isclose
def ai_to_reflectivity(ai,win=7,threshold=8e-4):
    '''
    Acoustic Impedance to Reflectivity
    '''    
    # compute reflectivity coeff
    refl=np.zeros_like(ai)
    for i in range(len(ai)-1):
        if isclose((ai[i+1]+ai[i]), 0): continue
        R=(ai[i+1]-ai[i])/(ai[i+1]+ai[i])
        refl[i+1]=R
      
    ind=[]
    for i in range(win,len(ai)-win):
        for kk in range(1,win):
            if (np.abs(refl[i]-refl[i-kk])<threshold) or (np.abs(refl[i]-refl[i+kk])<threshold):
                ind.append(i)
                break     
    refl[ind]=0
    
    slope=np.zeros(np.shape(ai))
    ind=np.where(np.abs(refl)>0)[0]
    for iii in range(len(slope)-1):
        slope[iii+1]=ai[iii+1]-ai[iii]
    slope[ind]=0
    
    # for iii in range(0,len(ind)-1):
    #     if ind[iii+1]-ind[iii]>1:
    #         slope[ind[iii]+1:ind[iii+1]] = ai[ind[iii]+1]-ai[ind[iii]]
    return refl,slope

def _ai_to_reflectivity(ai,win=7,threshold=8e-4):
    '''
    Acoustic Impedance to Reflectivity
    '''
    ai = np.array(ai)
    if len(ai.shape) == 1: ai.reshape((1, *ai.shape))
    refl = (ai[:, 1:]-ai[:, :-1])/(ai[:, 1:]+ai[:, :-1])
    refl = np.pad(refl, (1, 0), mode='constant')
    
    ind = []
    for i in range(win, ai.shape[1]): # All moving window positions
        if np.any(np.abs(refl[:, i]-refl[:, (i-win):i])<threshold):
            ind.append(i)
    
    slope=np.zeros(np.shape(ai))
    ind=np.where(np.abs(refl)>0)[0]
    slope[:, 1:] = np.diff(ai, axis=1)
    slope[ind]=0
    
    return refl, slope
  
    
def oversample_refl(t,refl,slope,dtnew):
    dt=t[1]-t[0]
    tnew=np.arange(t[0],t[-1]+dtnew,dtnew) 
    factor=dt/dtnew
    
    iref=np.where(np.abs(refl)>0)[0]
    irefnew=np.array(iref*factor,dtype=int)
    reflnew=np.zeros(len(tnew))
    reflnew[irefnew]=refl[iref]
    
    slopenew=np.zeros(len(tnew))
    for i in range(0,len(iref)-1): 
        if len(slope[iref[i]+1:iref[i+1]])>0:
            S=np.sum(slope[iref[i]+1:iref[i+1]])
            D=iref[i+1]*4-iref[i]*4-1
            slopenew[iref[i]*4+1:iref[i+1]*4]=S/D
    
    return reflnew, slopenew, tnew



##############################################
#### Function for wavelets
def ricker_wavelet(fc=25, duration=None, dt=1/1000, plt_flag=False):
    '''
    position is center - for convolution - or start - for JP code
    dt in [sec]
    duration [sec] - if none, then 1/(2fc)
    '''
    if duration == None:
        duration=2/fc
    
    t = np.arange(-duration/2, (duration+dt)/2, dt)
    wp=2*np.pi*fc
    
    if duration is None:
        source = (1-1/2*wp**2*t**2)*np.exp(-1/4*wp**2*(t**2))
    else:    
        t=t-np.min(t)
        coef=1
        tb=1/(fc*coef)
        source=(1-1/2*wp**2*(t-tb)**2)*np.exp(-1/4*wp**2*(t-tb)**2);
    
    if plt_flag:
        plt.figure()
        plt.plot(t,source)
    return t,source


def resample_seis(tover,seis_over,t):
    # resampling to the Marmousi dt
    f=interp1d(tover,seis_over,kind='linear')
    seis_model=f(t)
    return seis_model




