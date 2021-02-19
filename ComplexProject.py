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
import re
import math as m

L = 16                  ##################
P = 0.5                 #input parameters#
RunTime = 100000        ##################
Systems = np.array([4,8,16,32,64,128,256])

SteadyState = False             #################
slopes = np.zeros(L)            #initialisations#
thresholds = np.zeros(L)        #################

changes = []    #keeps track of which sites relaxed
totH = 0        #height of the pile
totT = 0        #total time since the beginning
pileH = []      #list for pile height at times totT
tc = []         #records time after ss has been reached
s = 0           #avalnche size
s_sizes = []    #list for avalanche sizes

for i in range(len(thresholds)):             #randomise thresholds
    thresholds[i] = np.random.choice([1,2], p=[P,1-P])
    

def Drive(GrainNo = 1):
    '''Drives the system by adding number of grains determined by GrainNo'''
    global totH
    global totT
    slopes[0] = slopes[0] + GrainNo
    totH += GrainNo
    totT += 1

def RelaxFirst():
    '''Checks whether first site needs to be relaxed and if so, relaxes it'''
    global totH
    global s
    if slopes[0]>thresholds[0]:
        slopes[0] -= 2
        slopes[1] += 1
        thresholds[0] = np.random.choice([1,2], p=[P,1-P])
        changes.append(0)
        changes.append(1)
        totH -= 1
        s += 1


def RelaxSelect(L=4, a=[]):
    '''Checks if all the sites in the sites to be relaxed list need relaxation,
    and if so, relaxes them then outputs the updated list of sites to check'''
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
                totH -= 1
                s += 1
        elif a[i] < L-1:
            if slopes[a[i]]>thresholds[a[i]]:
                slopes[a[i]] -= 2
                slopes[a[i]+1] += 1
                slopes[a[i]-1] += 1
                thresholds[a[i]] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(a[i])
                newChanges.append(a[i]+1)
                newChanges.append(a[i]-1)
                s += 1
        else:
            if slopes[L-1]>thresholds[L-1]:
                slopes[L-1] -= 1
                slopes[L-2] += 1
                thresholds[L-1] = np.random.choice([1,2], p=[P,1-P])
                newChanges.append(L-1)
                newChanges.append(L-2)
                SteadyState = True
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
            changes = list(dict.fromkeys(RelaxSelect(L, changes))) 
        if SteadyState == True:
            #break
            #tc.append(totT)
            #pileH.append(totH)
            s_sizes.append(s)
        tc.append(totT)
        pileH.append(totH)
        s = 0
#%%        
'''
Task 1
'''
L = 16
RunTime = 100000
u = []
var = []
for i in range(0,10):
    Run(L, RunTime)
    u.append(sum(pileH)/len(pileH))
    var.append((np.var(pileH)))
    
    slopes = np.zeros(L)
    thresholds = np.zeros(L)
    changes = []
    pileH = []
    totH = 0
    totT = 0
    for i in range(len(thresholds)):
        thresholds[i] = np.random.choice([1,2], p=[P,1-P])

print('The average height is:', sum(u)/len(u), '+-', np.sqrt(sum(var)/len(var)))
#%%
'''
Task 2a
'''
datax = {}
datay = {}
RunTime = 100000
for i in Systems:
    pileH = []
    totT = 0
    totH = 0
    changes = []
    SteadyState = False             #################
    slopes = np.zeros(i)            #initialisations#
    thresholds = np.zeros(i)        #################
    for k in range(len(thresholds)):             #randomise thresholds
        thresholds[k] = np.random.choice([1,2], p=[P,1-P])

    Run(i, RunTime)
    datax['x{0}'.format(i)] = list(range(totT))
    datay['y{0}'.format(i)] = pileH

with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2aDatax', 'wb') as f:
    pickle.dump(datax,f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2aDatay', 'wb') as f:
    pickle.dump(datay,f)
#%%
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2aDatax', 'rb') as f:
    datax = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2aDatay', 'rb') as f:
    datay = pickle.load(f)
plt.plot(datax['x4'],datay['y4'], label = 'L = 4')
plt.plot(datax['x8'],datay['y8'], label = 'L = 8')
plt.plot(datax['x16'],datay['y16'], label = 'L = 16')
plt.plot(datax['x32'],datay['y32'], label = 'L = 32')
plt.plot(datax['x64'],datay['y64'], label = 'L = 64')
plt.plot(datax['x128'],datay['y128'], label = 'L = 128')
plt.plot(datax['x256'],datay['y256'], label = 'L = 256')
plt.xlabel('$t$', fontsize = 15)
plt.ylabel('$h(t;L)$', fontsize = 15)
plt.legend(loc = 'upper left', fontsize = 10)
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2a',dpi=500)

#%%
'''
Task2b/c data
'''
times = []         #list of times to reach ss
iterations = []    #for a number of runs
RunTime = 100000
for i in range(10):
    for i in Systems:
        pileH = []
        totT = 0
        totH = 0
        changes = []
        SteadyState = False             #################
        slopes = np.zeros(i)            #initialisations#
        thresholds = np.zeros(i)        #################
        for k in range(len(thresholds)):             #randomise thresholds
            thresholds[k] = np.random.choice([1,2], p=[P,1-P])

        Run(i, RunTime)
        times.append(totT)
    iterations.append(times)
    times = []
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2bcData2', 'wb') as f:
    pickle.dump(iterations,f)
#%%

with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2bcData', 'rb') as f:
    iterations = pickle.load(f)

#iterations = [[18, 62, 210, 872, 3549, 14059, 55030], [18, 63, 221, 866, 3452, 14032, 55879], [14, 58, 224, 869, 3432, 13671, 55431], [12, 65, 215, 908, 3528, 13767, 56070], [13, 46, 221, 885, 3510, 13582, 55581], [20, 64, 206, 885, 3458, 13958, 56655], [18, 62, 231, 899, 3469, 13626, 56450], [13, 62, 226, 934, 3376, 14098, 56455], [12, 61, 215, 913, 3638, 14334, 56140], [17, 51, 213, 893, 3469, 13738, 55873]]
avg = []
for i in range(len(iterations[0])):
    a = []
    for k in range(len(iterations)):  
        a.append(iterations[k][i])
    avg.append(sum(a)/len(a))

def Fit(L, a, b):
    H = a*(L**b)
    return H

popt, pcov = cf(Fit, Systems, avg)

print(popt,pcov)

plt.plot(Systems, avg, '--bo', label = 'data')
plt.plot(Systems, Fit(Systems,*popt), '-r', label = 'fit')
plt.xlabel('$L$', fontsize=15)
plt.ylabel(r'$\langle t_{c}(L) \rangle$', fontsize=15)
plt.legend(fontsize=10)
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2b',dpi=500)

#%%
'''
Task 2d data
'''
datax = {}
datay = {}
RunTime = 100000
for i in Systems:
    heights = []       #list of heights on iterations
    for k in range(10):
        totT = 0
        changes = []
        pileH = []
        tc = []
        SteadyState = False             #################
        slopes = np.zeros(i)            #initialisations#
        thresholds = np.zeros(i)        #################
        for k in range(len(thresholds)):             #randomise thresholds
            thresholds[k] = np.random.choice([1,2], p=[P,1-P])

        Run(i, RunTime)             
        heights.append(pileH)
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

with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2dDatax2', 'wb') as f:
    pickle.dump(datax,f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2dDatay2', 'wb') as f:
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
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t/L^{2}$',fontsize=15)
plt.ylabel(r'$\tilde{h}/L$', fontsize=15)
plt.legend(loc = 'lower right', fontsize=10)
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2d', dpi=500)

#%%
'''
tasks 2e-2g data
'''
heights = []
heightsSquare = []
heightProbs = {}
RunTime = 1000000
for i in Systems:
    totT = 0
    changes = []
    pileH = []
    tc = []
    SteadyState = False             #################
    slopes = np.zeros(i)            #initialisations#
    thresholds = np.zeros(i)        #################
    for k in range(len(thresholds)):             #randomise thresholds
        thresholds[k] = np.random.choice([1,2], p=[P,1-P])

    Run(i, RunTime)             
    heights.append(sum(pileH)/(tc[-1]-tc[0]))
    heightsSquare.append((sum(map(lambda x: x**2, pileH)))/(tc[-1]-tc[0]))
    for j in range(int(min(pileH)), int(max(pileH))+1):
        heightProbs['L{}-H{}'.format(i,j)]= pileH.count(j)  
        
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

#def Corr2(L,a0,a1,a2,w1,w2):
#    H = a0*L - a0*a1*(L**(1-w1)) + a0*a1*a2*(L**(1-w2))
#    return H

popt, pcov = cf(Corr, Systems, heights)
#popt2, pcov2 = cf(Corr2, Systems, heights)

print(popt, np.sqrt(pcov))
#print(popt2, pcov2)
plt.plot(Systems, heights, '--bo', label = 'data')
plt.plot(Systems, Corr(Systems,*popt), '-r', label = 'two term fit')
#plt.plot(Systems, Corr2(Systems,*popt2), '-g', label = 'three term fit')
plt.xlabel('$L$', fontsize = 15)
plt.ylabel(r'$\langle h(t;L) \rangle_{t}$', fontsize = 15)
plt.legend(fontsize = 10)
plt.show()
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2e', dpi=500)

#%%
'''
task 2f
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsRun3', 'rb') as f:
   heights = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsSquaredRun3', 'rb') as f:
   heightsSquared = pickle.load(f)
   
stdevs = []
for k in range(len(heights)):
    stdevs.append(np.sqrt(heightsSquared[k] - heights[k]**2))

def Fit(L, a, b):
    R = a*L**b
    return R

popt, pcov = cf(Fit, Systems, stdevs)
print(popt, np.sqrt(pcov))
plt.plot(Systems, stdevs, 'bo', label = 'data')
plt.plot(Systems, Fit(Systems, *popt), '-r', label = 'fit')
plt.xlabel('$L$', fontsize = 15)
plt.ylabel(r'$\sigma_{h}(L)$', fontsize = 15)
plt.legend(fontsize = 10)
plt.show()
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2f', dpi=500)

   
#%%
'''
task 2ga
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightProbsRun3', 'rb') as f:
   heightProbs = pickle.load(f)

H = globals()
C = globals()

for i in Systems:
    H['Heights{0}'.format(i)] = [value for key, value in heightProbs.items() if 'L{0}'.format(i) in key]
    H['Heights{0}'.format(i)] = [x/sum(H['Heights{0}'.format(i)]) for x in H['Heights{0}'.format(i)]]
    C['Config{0}'.format(i)] = [key for key, value in heightProbs.items() if 'L{0}'.format(i) in key]
    C['Config{0}'.format(i)] = [int(re.sub('L{0}-H'.format(i),'',x)) for x in C['Config{0}'.format(i)]]
    

plt.plot(Config4, Heights4, label = 'L = 4')
plt.plot(Config8, Heights8, label = 'L = 8')
plt.plot(Config16, Heights16, label = 'L = 16')
plt.plot(Config32, Heights32, label = 'L = 32')
plt.plot(Config64, Heights64, label = 'L = 64')
plt.plot(Config128, Heights128, label = 'L = 128')
plt.plot(Config256, Heights256, label = 'L = 256')
plt.legend(fontsize = 10)
plt.xlabel('$h$', fontsize = 15)
plt.ylabel(r'$P(h,L)$', fontsize = 15)
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2ga', dpi=500)
#%%
'''
task 2gb 
'''

with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsRun3', 'rb') as f:
   heights = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightsSquaredRun3', 'rb') as f:
   heightsSquared = pickle.load(f)
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2eHeightProbsRun3', 'rb') as f:
   heightProbs = pickle.load(f)
   
stdevs = [np.sqrt(heightsSquared[x]-heights[x]**2) for x in range(7)]

H = globals()
C = globals()

for i in Systems:
    H['Heights{0}'.format(i)] = [value for key, value in heightProbs.items() if 'L{0}'.format(i) in key]
    H['Heights{0}'.format(i)] = [x*(stdevs[int(m.log(i,2)-2)])/sum(H['Heights{0}'.format(i)]) for x in H['Heights{0}'.format(i)]]
    C['Config{0}'.format(i)] = [key for key, value in heightProbs.items() if 'L{0}'.format(i) in key]
    C['Config{0}'.format(i)] = [int(re.sub('L{0}-H'.format(i),'',x)) for x in C['Config{0}'.format(i)]]
    C['Config{0}'.format(i)] = [(x-heights[int(m.log(i,2)-2)])/stdevs[int(m.log(i,2)-2)] for x in C['Config{0}'.format(i)]]

   
plt.plot(Config4, Heights4, '+', label = 'L = 4')
plt.plot(Config8, Heights8, '+', label = 'L = 8')
plt.plot(Config16, Heights16, '+', label = 'L = 16')
plt.plot(Config32, Heights32, '+', label = 'L = 32')
plt.plot(Config64, Heights64, '+', label = 'L = 64')
plt.plot(Config128, Heights128, '+', label = 'L = 128')
plt.plot(Config256, Heights256, '+', label = 'L = 256')
plt.legend(fontsize = 10)
plt.xlabel(r'$(h - \langle h \rangle ) / \sigma$', fontsize = 15)
plt.ylabel(r'$\sigma_h P(h,L)$', fontsize = 15)
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/2gb', dpi=500)

#%%
'''
task 3 data
'''
aval_size_data = {}
RunTime = 1000000
for i in Systems:
    totT = 0
    changes = []
    pileH = []
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
    
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3aval2', 'wb') as f:
    pickle.dump(aval_size_data,f) 
#%%
'''
3a    
'''
with open('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3aval', 'rb') as f:
     aval_size_data = pickle.load(f)

ocurrences = {}
for i in Systems:
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
for k in [1,2,3,4]:
    for i in Systems:
        m = []
        for s in aval_size_data['L{}'.format(i)]:
            m.append(s**k)
        moments[k-1].append(sum(m)/len(m))
        
plt.plot(Systems, moments[0], '--bo', label = '$k=1$')
plt.plot(Systems, moments[1], '--ro', label = '$k=2$')
plt.plot(Systems, moments[2], '--mo', label = '$k=3$')
plt.plot(Systems, moments[3], '--go', label = '$k=4$')
plt.yscale('log')
#plt.xscale('log')
plt.legend()
plt.xlabel('$L$', fontsize = 15)
plt.ylabel(r'$\langle s^{k} \rangle$', fontsize = 15)
plt.savefig('/home/adam/Desktop/work/3rd Year/Complexity&Networks/Oslo model project/3b',dpi=500)

    

        
    