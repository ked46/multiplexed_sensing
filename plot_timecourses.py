'''Define functions to import and process the files listed above'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plottimecourselist(datalist,time_vector,ax):
    '''plot each experiment plate data in different colors for that experiment'''
    colors = ['r', 'b', 'g', 'y']  # Define colors for different readers
    for i,(num,array) in enumerate(datalist.items()):
        color=colors[i]
        for row in (datalist[i]):
            ax.plot(time_vector,row,color=color,alpha=0.5)
            
def plottimecoursearray(array,time_vector,ax):
    for row in (array):
        ax.plot(time_vector,row,alpha=0.5)
    

def figure_layout(sensors,rawlists,convlists,subtractedarrays,normalizedarrays,time_vector,fluors):
    # Reset to Matplotlib defaults
    plt.style.use('default')  # Ensures Matplotlib is fully reset
    # Set Matplotlib to keep text as text in SVG files
    plt.rcParams['svg.fonttype'] = 'none'
    fig, axes=plt.subplots(sensors+1,4,figsize=(13,(sensors+1)*2))
    
    for i in range(sensors+1):
        ax=axes[i,0]
        plottimecourselist(rawlists[i],time_vector,ax)
        if i==0:
            ax.set_ylabel('OD',fontsize=12)
        else:
            ax.set_ylabel(f'{fluors[i-1]}',fontsize=12)
    for i in range(sensors+1):
        ax=axes[i,1]
        plottimecourselist(convlists[i],time_vector,ax)
    for i in range(sensors):
        ax=axes[i+1,2]
        plottimecoursearray(subtractedarrays[i],time_vector,ax)
    for i in range(sensors):
        ax=axes[i+1,3]
        plottimecoursearray(normalizedarrays[i],time_vector,ax)
    axes[0,0].set_title('raw data',fontsize=12)
    axes[0,1].set_title('plate reader converted data',fontsize=12)
    axes[0,2].set_title('basal expression subtracted data',fontsize=12)
    axes[0,3].set_title('normalized data',fontsize=12)
    axes[sensors,0].set_xlabel('Time (hr)',fontsize=12)
