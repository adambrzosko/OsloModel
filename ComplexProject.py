#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:11:28 2021

@author: adam
"""

import numpy as np
L = 16
P = 0.5
SteadyState = 1000
RunTime = 10000

slopes = np.zeros(L)
thresholds = np.zeros(L)
changes = []
h1 = []
for i in range(len(thresholds)):
    thresholds[i] = np.random.choice([1,2], p=[P,1-P])
    

def Drive(GrainNo = 1):
    slopes[0] = slopes[0] + GrainNo

def RelaxAll():
    if slopes[0]>thresholds[0]:
        slopes[0] -= 2
        slopes[1] += 1
        thresholds[0] = np.random.choice([1,2], p=[P,1-P])
        changes.append(0)
        changes.append(1)
    for i in range(1,L-1):
        if slopes[i]>thresholds[i]:
            slopes[i] -= 2
            slopes[i+1] += 1
            slopes[i-1] += 1
            thresholds[i] = np.random.choice([1,2], p=[P,1-P])
            changes.append(i)
            changes.append(i-1)
            changes.append(i+1)
    if slopes[L-1]>thresholds[L-1]:
        slopes[L-1] -= 1
        slopes[L-2] += 1
        thresholds[L-1] = np.random.choice([1,2], p=[P,1-P])
        changes.append(L-1)
        changes.append(L-2)

def RelaxSelect(a=[]):
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
    return newChanges
    
def Run(relPeriods = 1000):
    global changes
    for i in range(relPeriods):
        Drive()
        RelaxAll()
        changes = list(dict.fromkeys(changes)) #gets rid of repeats
        #print('C',changes)
        while changes != []:
            changes = list(dict.fromkeys(RelaxSelect(changes))) #gets rid of repeats
            #print('CC',changes)
            #print('SS',slopes)
            #print('T',thresholds)
        if RunTime > SteadyState:
            h1.append(sum(slopes))
        #print(h1)
        #print('S',slopes)

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


