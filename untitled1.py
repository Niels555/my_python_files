# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:01:32 2019

@author: jonkm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import math

a=np.array([0,1,2])
b = np.array([2,2,2])
minimum = np.min(a)
print(minimum)

n=1
if a.all() == b.all():
    n=4
print(n)
c=np.array([0,-1,2,3,-4,5])
d = c[0:6]
print(d)
e=c[0:1]
print(e)
print(abs(c))
c+=[8]
print(c)
print(c[0:1])
array = np.zeros(1)
freek = np.array([1,2,3,4])
print(np.sum(freek[1:4]))

print(np.ones(1))
print(np.ones(1)*0.5*1)


