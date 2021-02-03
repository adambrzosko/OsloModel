#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:11:28 2021

@author: adam
"""

import numpy as np
import matplotlib.pyplot as plt

L = 16                  ##################
P = 0.5                 #input parameters#
RunTime = 70000         ##################

SteadyState = False             #################
slopes = np.zeros(L)            #initialisations#
thresholds = np.zeros(L)        #################

changes = []    #keeps track of which sites relaxed
h1 = []         #height of site 1 
totH = 0        #total sum of heights of all site
totT = 0        #total time since the beginning
pileH = []      #list for total heights at times totT
tc = []         #records time after ss has been reached

for i in range(len(thresholds)):             #randomise thresholds
    thresholds[i] = np.random.choice([1,2], p=[P,1-P])
    

def Drive(GrainNo = 1):
    global totH
    global totT
    slopes[0] = slopes[0] + GrainNo
    totH += GrainNo
    totT += 1

def RelaxFirst():
    if slopes[0]>thresholds[0]:
        slopes[0] -= 2
        slopes[1] += 1
        thresholds[0] = np.random.choice([1,2], p=[P,1-P])
        changes.append(0)
        changes.append(1)


def RelaxSelect(L=4, a=[]):
    global totH
    global SteadyState
    newChanges = []
    for i in range(0,len(a)):
        if a[i] == 0:
            if slopes[0]>thresholds[0]:
                slopes[0] -= 2
                slopes[1] += 1
                thresholds[0] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(0)
                newChanges.append(1)
        if a[i] != 0 and a[i] < L-1:
            if slopes[a[i]]>thresholds[a[i]]:
                slopes[a[i]] -= 2
                slopes[a[i]+1] += 1
                slopes[a[i]-1] += 1
                thresholds[a[i]] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(a[i])
                newChanges.append(a[i]+1)
                newChanges.append(a[i]-1)
        if a[i] == L-1:
            if slopes[L-1]>thresholds[L-1]:
                slopes[L-1] -= 1
                slopes[L-2] += 1
                thresholds[L-1] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(L-1)
                newChanges.append(L-2)
                SteadyState = True
                totH -= 1
    return newChanges
    
def Run(L, relPeriods = 1000):
    '''Executes the simulation with standard #drives = 1000'''
    global totH
    global changes
    global slopes
    for i in range(relPeriods):
        Drive()
        RelaxFirst()
        changes = list(dict.fromkeys(changes)) #gets rid of repeats
        while changes != []:
            changes = list(dict.fromkeys(RelaxSelect(L, changes))) #gets rid of repeats
        h1.append(sum(slopes))
        if SteadyState == True:
         #   break
         tc.append(totT)
        pileH.append(totH)
#%%        
'''
Task 1
'''
u = []
var = []
for i in range(0,10):
    Run(RunTime)
    u.append(sum(h1)/len(h1))
    var.append((np.var(h1)))
    
    slopes = np.zeros(L)
    thresholds = np.zeros(L)
    changes = []
    h1 = []
    for i in range(len(thresholds)):
        thresholds[i] = np.random.choice([1,2], p=[P,1-P])

print('The average height is:', sum(u)/len(u), '+-', np.sqrt(sum(var)/len(var)))
#%%
'''
Task 2a
'''
datax = {}
datay = {}
for i in [4, 8, 16, 32, 64, 128, 256]:
    pileH = []
    totT = 0
    totH = 0
    changes = []
    h1 = []
    SteadyState = False             #################
    slopes = np.zeros(i)            #initialisations#
    thresholds = np.zeros(i)        #################
    for k in range(len(thresholds)):             #randomise thresholds
        thresholds[k] = np.random.choice([1,2], p=[P,1-P])

    Run(i, RunTime)
    datax['x{0}'.format(i)] = list(range(totT))
    datay['y{0}'.format(i)] = h1

plt.plot(datax['x4'],datay['y4'], label = 'L = 4')
plt.plot(datax['x8'],datay['y8'], label = 'L = 8')
plt.plot(datax['x16'],datay['y16'], label = 'L = 16')
plt.plot(datax['x32'],datay['y32'], label = 'L = 32')
plt.plot(datax['x64'],datay['y64'], label = 'L = 64')
plt.plot(datax['x128'],datay['y128'], label = 'L = 128')
plt.plot(datax['x256'],datay['y256'], label = 'L = 256')
plt.title('Time to reach steady state')
plt.xlabel('Time since beginning of simulation')
plt.ylabel('Height of pile')
plt.legend(loc = 'upper left')

#%%
'''
Task2b/c
'''
times = []         #list of times to reach ss
iterations = []    #for a number of runs
for i in range(10):
    for i in [4, 8, 16, 32, 64, 128, 256]:
        pileH = []
        totT = 0
        totH = 0
        changes = []
        h1 = []
        SteadyState = False             #################
        slopes = np.zeros(i)            #initialisations#
        thresholds = np.zeros(i)        #################
        for k in range(len(thresholds)):             #randomise thresholds
            thresholds[k] = np.random.choice([1,2], p=[P,1-P])

        Run(i, RunTime)
        times.append(totT)
    iterations.append(times)
    times = []
print(iterations)
#%%
'''
Task 2d
'''
datax = {}
datay = {}

for i in [4, 8, 16, 32, 64, 128, 256]:
    tcs = []           #list of crossover times for scaling
    heights = []       #list of heights on iterations
    for k in range(10):
        totT = 0
        changes = []
        h1 = []
        tc = []
        SteadyState = False             #################
        slopes = np.zeros(i)            #initialisations#
        thresholds = np.zeros(i)        #################
        for k in range(len(thresholds)):             #randomise thresholds
            thresholds[k] = np.random.choice([1,2], p=[P,1-P])

        Run(i, RunTime)             
        tcs.append(tc[0])
        heights.append(h1)
    t_c = np.mean(tcs)
    scaled_t = [x/t_c for x in list(range(RunTime))]
    scaled_h = [x/i for x in np.mean(heights, axis = 0)]
    datax['x{0}'.format(i)] = scaled_t
    datay['y{0}'.format(i)] = scaled_h
        
    '''like averaging graphs then scaling them, will have a think how (h/L vs t/tc)'''
datax['x4'].pop(0)
datay['y4'].pop(0)
datax['x8'].pop(0)
datay['y8'].pop(0)
datay['y16'].pop(0)
datax['x16'].pop(0)
datax['x32'].pop(0)
datay['y32'].pop(0)
datay['y64'].pop(0)
datax['x64'].pop(0)
datax['x128'].pop(0)
datay['y128'].pop(0)
datay['y256'].pop(0)
datax['x256'].pop(0)
    
plt.plot(np.log(datax['x4']),np.log(datay['y4']), label = 'L = 4')
plt.plot(np.log(datax['x8']),np.log(datay['y8']), label = 'L = 8')
plt.plot(np.log(datax['x16']),np.log(datay['y16']), label = 'L = 16')
plt.plot(np.log(datax['x32']),np.log(datay['y32']), label = 'L = 32')
plt.plot(np.log(datax['x64']),np.log(datay['y64']), label = 'L = 64')
plt.plot(np.log(datax['x128']),np.log(datay['y128']), label = 'L = 128')
plt.plot(np.log(datax['x256']),np.log(datay['y256']), label = 'L = 256')
plt.title('Data collapse for the scaled height')
plt.xlabel('Time/cross-over time')
plt.ylabel('Height/system size')
plt.legend(loc = 'upper left')
