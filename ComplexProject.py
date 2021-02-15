#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:11:28 2021

@author: adam
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit as cf

L = 16                  ##################
P = 0.5                 #input parameters#
RunTime = 100000        ##################

SteadyState = False             #################
slopes = np.zeros(L)            #initialisations#
thresholds = np.zeros(L)        #################

changes = []    #keeps track of which sites relaxed
h1 = []         #height of site 1 
totH = 0        #total sum of heights of all site
totT = 0        #total time since the beginning
pileH = []      #list for total heights at times totT
tc = []         #records time after ss has been reached
s = 0           #avalnche size
s_sizes = []    #list for avalanche sizes

for i in range(len(thresholds)):             #randomise thresholds
    thresholds[i] = np.random.choice([1,2], p=[P,1-P])
    

def Drive(GrainNo = 1):
    global totH
    global totT
    slopes[0] = slopes[0] + GrainNo
    totH += GrainNo
    totT += 1

def RelaxFirst():
    global s
    if slopes[0]>thresholds[0]:
        slopes[0] -= 2
        slopes[1] += 1
        thresholds[0] = np.random.choice([1,2], p=[P,1-P])
        changes.append(0)
        changes.append(1)
        s += 1


def RelaxSelect(L=4, a=[]):
    global totH
    global SteadyState
    global s
    newChanges = []
    for i in range(0,len(a)):
        if a[i] == 0:
            if slopes[0]>thresholds[0]:
                slopes[0] -= 2
                slopes[1] += 1
                thresholds[0] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(0)
                newChanges.append(1)
                s += 1
        if a[i] != 0 and a[i] < L-1:
            if slopes[a[i]]>thresholds[a[i]]:
                slopes[a[i]] -= 2
                slopes[a[i]+1] += 1
                slopes[a[i]-1] += 1
                thresholds[a[i]] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(a[i])
                newChanges.append(a[i]+1)
                newChanges.append(a[i]-1)
                s += 1
        if a[i] == L-1:
            if slopes[L-1]>thresholds[L-1]:
                slopes[L-1] -= 1
                slopes[L-2] += 1
                thresholds[L-1] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(L-1)
                newChanges.append(L-2)
                SteadyState = True
                totH -= 1
                s += 1
    return newChanges
    
def Run(L, relPeriods = 1000):
    '''Executes the simulation with standard #drives = 1000'''
    global totH
    global changes
    global slopes
    global s
    for i in range(relPeriods):
        Drive()
        RelaxFirst()
        changes = list(dict.fromkeys(changes)) #gets rid of repeats
        while changes != []:
            changes = list(dict.fromkeys(RelaxSelect(L, changes))) #gets rid of repeats
        #h1.append(sum(slopes))
        if SteadyState == True:
            #   break
            tc.append(totT)
            h1.append(sum(slopes))
            s_sizes.append(s)
        pileH.append(totH)
        s = 0
#%%        
'''
Task 1
'''
for k in [16,32]:
    u = []
    var = []
    for i in range(0,10):
        Run(k, RunTime)
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
plt.xlabel('Time since beginning of simulation', fontsize = 20)
plt.ylabel('Height of pile', fontsize = 20)
plt.legend(loc = 'upper left')
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2a',dpi=500)

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
Task 2d data
'''
datax = {}
datay = {}

for i in [4, 8, 16, 32, 64, 128, 256]:
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
        heights.append(h1)
    scaled_t = [x/(i**2) for x in list(range(RunTime))]
    scaled_h = [x/i for x in np.mean(heights, axis = 0)]
    datax['x{0}'.format(i)] = scaled_t
    datay['y{0}'.format(i)] = scaled_h
        
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

with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2dDatax', 'wb') as f:
    pickle.dump(datax,f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2dDatay', 'wb') as f:
    pickle.dump(datay,f)
#%%
'''
2d plot

'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2dDatax', 'rb') as f:
    datax = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2dDatay', 'rb') as f:
    datay = pickle.load(f)
    
plt.plot(datax['x4'],datay['y4'], label = 'L = 4')
plt.plot(datax['x8'],datay['y8'], label = 'L = 8')
plt.plot(datax['x16'],datay['y16'], label = 'L = 16')
plt.plot(datax['x32'],datay['y32'], label = 'L = 32')
plt.plot(datax['x64'],datay['y64'], label = 'L = 64')
plt.plot(datax['x128'],datay['y128'], label = 'L = 128')
plt.plot(datax['x256'],datay['y256'], label = 'L = 256')
plt.title('Data collapse for the scaled height')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Time/(system size)^2')
plt.ylabel('Height/system size')
plt.legend(loc = 'upper left')

#%%
'''
tasks 2e-2g data
'''
heights = []
heightsSquare = []
heightProbs = {}
for i in [4, 8, 16, 32, 64, 128, 256]:
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
    heights.append(sum(h1)/(tc[-1]-tc[0]))
    heightsSquare.append((sum(map(lambda x: x**2, h1)))/(tc[-1]-tc[0]))
    for j in range(int(min(h1)), int(max(h1))+1):
        heightProbs['L{}-H{}'.format(i,j)]= h1.count(j)  
        
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsRun3', 'wb') as f:
    pickle.dump(heights,f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsSquaredRun3', 'wb') as f:
    pickle.dump(heightsSquare,f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightProbsRun3', 'wb') as f:
    pickle.dump(heightProbs,f)
    
#%%
'''
task 2e
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsRun3', 'rb') as f:
   heights = pickle.load(f)

def Corr(L, a0, a1, w1):
    H = a0*L - a0*a1*(L**(1-w1))
    return H

S = np.array([4, 8, 16, 32, 64, 128, 256])
popt, pcov = cf(Corr, S, heights)

print(popt, pcov)
plt.plot(S, heights, '--bo', label = 'data')
plt.plot(S, Corr(S,*popt), '-r', label = 'fit')
plt.xlabel('System size')
plt.ylabel('Average height')
plt.title('Time-averaged heights against system size in steady state')
plt.legend()
plt.show()

#%%
'''
task 2f
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsRun3', 'rb') as f:
   heights = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsSquaredRun3', 'rb') as f:
   heightsSquared = pickle.load(f)
   
S = np.array([4, 8, 16, 32, 64, 128, 256])
stdevs = []
for k in range(len(heights)):
    stdevs.append(np.sqrt(heightsSquared[k] - heights[k]**2))
plt.plot(S, stdevs, '--bo', label = 'data')
plt.xlabel('System size')
plt.ylabel('Average height')
plt.title('Standard deviation of average heights against system size')
plt.legend()
plt.show()
   
#%%
'''
task 2ga
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightProbsRun2', 'rb') as f:
   heightProbs = pickle.load(f)

Heights4 = []
Config4 = []
for i in range(4,9):
    Heights4.append(heightProbs['L4-H{}'.format(i)])
    Config4.append(i)
Heights4 = [x/sum(Heights4) for x in Heights4]

Heights8 = []
Config8 = []
for i in range(10,17):
    Heights8.append(heightProbs['L8-H{}'.format(i)])
    Config8.append(i)
Heights8 = [x/sum(Heights8) for x in Heights8]
    
Heights16 = []
Config16 = []
for i in range(22,33):
    Heights16.append(heightProbs['L16-H{}'.format(i)])
    Config16.append(i)
Heights16 = [x/sum(Heights16) for x in Heights16]
    
Heights32 = []
Config32 = []
for i in range(49,62):
    Heights32.append(heightProbs['L32-H{}'.format(i)])
    Config32.append(i)
Heights32 = [x/sum(Heights32) for x in Heights32]
    
Heights64 = []
Config64 = []
for i in range(103,118):
    Heights64.append(heightProbs['L64-H{}'.format(i)])
    Config64.append(i)
Heights64 = [x/sum(Heights64) for x in Heights64]

Heights64 = [x/sum(Heights64) for x in Heights64]

Heights128 = []
Config128 = []
for i in range(212,229):
    Heights128.append(heightProbs['L128-H{}'.format(i)])
    Config128.append(i)
Heights128 = [x/sum(Heights128) for x in Heights128]
    
Heights256 = []
Config256 = []
for i in range(432,450):
    Heights256.append(heightProbs['L256-H{}'.format(i)])
    Config256.append(i)
Heights256 = [x/sum(Heights256) for x in Heights256]

plt.plot(Config4, Heights4, label = 'L = 4')
plt.plot(Config8, Heights8, label = 'L = 8')
plt.plot(Config16, Heights16, label = 'L = 16')
plt.plot(Config32, Heights32, label = 'L = 32')
plt.plot(Config64, Heights64, label = 'L = 64')
plt.plot(Config128, Heights128, label = 'L = 128')
plt.plot(Config256, Heights256, label = 'L = 256')
plt.legend()
plt.xlabel('h')
plt.ylabel(r'$P(h,L)$')
plt.title('Probability of configurations of given height')

#%%
'''
task 2gb
maybe multiply y by stdevand and (x-avgheight)/stdev  
'''

with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsRun2', 'rb') as f:
   heights = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eStdevsRun2', 'rb') as f:
   stdevs = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightProbsRun2', 'rb') as f:
   heightProbs = pickle.load(f)
   
Heights4 = []
Config4 = []
for i in range(4,9):
    Heights4.append(heightProbs['L4-H{}'.format(i)]*stdevs[0])
    Config4.append((i-heights[0])/stdevs[0])
Heights4 = [x/sum(Heights4) for x in Heights4]

Heights8 = []
Config8 = []
for i in range(10,17):
    Heights8.append(heightProbs['L8-H{}'.format(i)]*stdevs[1])
    Config8.append((i-heights[1])/stdevs[1])
Heights8 = [x/sum(Heights8) for x in Heights8]
    
Heights16 = []
Config16 = []
for i in range(22,33):
    Heights16.append(heightProbs['L16-H{}'.format(i)]*stdevs[2])
    Config16.append((i-heights[2])/stdevs[2])
Heights16 = [x/sum(Heights16) for x in Heights16]
    
Heights32 = []
Config32 = []
for i in range(49,62):
    Heights32.append(heightProbs['L32-H{}'.format(i)]*stdevs[3])
    Config32.append((i-heights[3])/stdevs[3])
Heights32 = [x/sum(Heights32) for x in Heights32]
   
Heights64 = []
Config64 = []
for i in range(103,118):
    Heights64.append(heightProbs['L64-H{}'.format(i)]*stdevs[4])
    Config64.append((i-heights[4])/stdevs[4])
Heights64 = [x/sum(Heights64) for x in Heights64]
    
Heights128 = []
Config128 = []
for i in range(212,229):
    Heights128.append(heightProbs['L128-H{}'.format(i)]*stdevs[5])
    Config128.append((i-heights[5])/stdevs[5])
Heights128 = [x/sum(Heights128) for x in Heights128]
    
Heights256 = []
Config256 = []
for i in range(432,450):
    Heights256.append(heightProbs['L256-H{}'.format(i)]*stdevs[6])
    Config256.append((i-heights[6])/stdevs[6])
Heights256 = [x/sum(Heights256) for x in Heights256]
   
plt.plot(Config4, Heights4, label = 'L = 4')
plt.plot(Config8, Heights8, label = 'L = 8')
plt.plot(Config16, Heights16, label = 'L = 16')
plt.plot(Config32, Heights32, label = 'L = 32')
plt.plot(Config64, Heights64, label = 'L = 64')
plt.plot(Config128, Heights128, label = 'L = 128')
plt.plot(Config256, Heights256, label = 'L = 256')
plt.legend()
plt.xlabel(r'$(h - \langle h \rangle ) / \sigma$')
plt.ylabel(r'$\sigma_h P(h,L)$')
plt.title('Data collapse for probability of configurations of given height')

#%%
'''
task 3 data
'''
aval_size_data = {}
for i in [4, 8, 16, 32, 64, 128, 256]:
    totT = 0
    changes = []
    h1 = []
    tc = []
    SteadyState = False             #################
    slopes = np.zeros(i)            #initialisations#
    thresholds = np.zeros(i)        #################
    s_sizes = []
    s = 0
    for k in range(len(thresholds)):             #randomise thresholds
        thresholds[k] = np.random.choice([1,2], p=[P,1-P])

    Run(i, RunTime)
    aval_size_data['L{0}'.format(i)] = s_sizes
    
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3aval', 'wb') as f:
    pickle.dump(aval_size_data,f) 
#%%
'''
3a    
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3aval', 'rb') as f:
     aval_size_data = pickle.load(f)

ocurrences = {}
for i in [4, 8, 16, 32, 64, 128, 256]:
    for k in range(0,max(aval_size_data['L{}'.format(i)])):
        ocurrences['L{}-{}'.format(i,k)] = aval_size_data['L{}'.format(i)].count(k)

    
#%%
'''
3a binned
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3aval', 'rb') as f:
     aval_size_data = pickle.load(f)

def logbin(data, scale = 2, zeros = False):
    """
    Taken from Max Falkenberg McGillivray
    mff113@ic.ac.uk
    2019 Complexity & Networks course 
    logbin230119.py v2.0
    23/01/2019
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y

x4, y4 = logbin(aval_size_data['L4'])
x8, y8 = logbin(aval_size_data['L8'])
x16, y16 = logbin(aval_size_data['L16'])
x32, y32 = logbin(aval_size_data['L32'])
x64, y64 = logbin(aval_size_data['L64'])
x128, y128 = logbin(aval_size_data['L128'])
x256, y256 = logbin(aval_size_data['L256'])
plt.plot(x4,y4, '--bo', label='L4')
plt.plot(x8,y8, '--co',label='L8')
plt.plot(x16,y16, '--go', label='L16')
plt.plot(x32,y32, '--ro',label='L32')
plt.plot(x64,y64, '--ko',label='L64')
plt.plot(x128,y128, '--yo',label='L128')
plt.plot(x256,y256, '--mo',label='L256')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Avalanche size')
plt.ylabel('Probability')
plt.title('Binned probabilities')
     
#%%
'''
3b
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3aval', 'rb') as f:
     aval_size_data = pickle.load(f)

moments = [[] for x in range(4)]
SysSize = [4, 8, 16, 32, 64, 128, 256]
for k in [1,2,3,4]:
    for i in SysSize:
        m = []
        for s in aval_size_data['L{}'.format(i)]:
            m.append(s**k)
        moments[k-1].append(sum(m)/len(m))
        
plt.plot(SysSize, moments[0], '--bo', label = '$k=1$')
plt.plot(SysSize, moments[1], '--ro', label = '$k=2$')
plt.plot(SysSize, moments[2], '--mo', label = '$k=3$')
plt.plot(SysSize, moments[3], '--go', label = '$k=4$')
plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.xlabel('System size L', fontsize = 15)
plt.ylabel(r'$\langle s^{k} \rangle$', fontsize = 15)
#plt.title('Avalanche size moments')
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/final',dpi=500)

    

        
    