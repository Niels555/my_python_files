# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:37:01 2019

@author: jonkm
"""

#building yield curve

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import math



def P(tau, DF,Tb): #function to make yieldcurve interpolation between spinepoints
    Num_payments = int(Tb/tau)          #number of DF's we are making
    P = np.zeros(Num_payments+1)        #we can go until y10, so 20 payments
    T = np.linspace(0,Tb,Num_payments+1)
    for i in range(0,Num_payments+1): #6 payments, then T0 until T6
        if i<=6:                        #this works for semi annual date up to and including y3
            if i%2 ==  1:                #this only works in the case we have three years and semi annual payments
                j = int(math.ceil(i/2))        
                P[i] = 0.5*(DF[j-1]+DF[j])  
            else: 
                j = int(i/2)
                P[i] = DF[j]
        if i>6 and i<=10:
            P[i]= (1- (T[i]-T[6])/(T[10]-T[6]))*DF[3]+((T[i]-T[6])/(T[10]-T[6]))*DF[4]      #linear interpolation, DF(3) is on T[6], DF(5) is on T[10])
        if i>10 and i <= 20:
            P[i] = (1- (T[i]-T[10])/(T[20]-T[10]))*DF[4]+((T[i]-T[10])/(T[20]-T[10]))*DF[5]  
    #elif tau == 1: 
    #    for i in range(0,Num_payments+1):
    #        P[i] = DF[i]     
    #else:
    #    print('tau was not 1 or 0.5')
    return P
            
def Swap(Ta,Tb,N,K,tau,P):             #first date, last date, notional, fixed rate in percentage, time between payments in years, Spine points
    Num_payments = int((Tb-Ta)/tau    )      #number of payments
    PtTi = P
    if tau ==1:    #annual payments, we need P[2], P[4], P[6] etc..
        Value = N*(PtTi[Num_payments*2]-PtTi[0])
        for i in range(1,Num_payments+1):
            j = i*2
            Value = Value + PtTi[j]*tau*K*0.01
    if tau ==0.5:
        Value = N*(PtTi[Num_payments]-PtTi[0])
        for i in range(1,Num_payments+1):
            Value = Value + PtTi[i]*tau*K*0.01  
    return Value

def NewSwap(Ta,Tb,N,K,tau,DF,t):    # t values at which we have the Discount Factors
    Num_payments = int((Tb-Ta)/tau)     #number of payments
    T = np.linspace(Ta,Tb,Num_payments+1)   #linspace at which to evaluate PtTi 
    interpol = np.interp(T,t,DF)        # interpolation by numpy, these are the PtTi
    print('interpol is',interpol)
    PtTi = np.zeros(len(T))             #length of PtTi vector is eqeal to T
    for i in range(0,Num_payments+1):   # for every Ti we will find left and right bounds for interpolation
        for j in range(0,len(t)):       #search through the t values of the spine points
            if T[i] <= t[j]:            #for the first tj Ti is smaller: this is index left bound, j+1 is then right bound
                lb = j-1
                rb = j
                break                   #breaks out of second loop, the right j is found  
        PtTi[i] = (1-(T[i]-t[lb])/(t[rb]-t[lb]))*DF[lb] + ((T[i]-t[lb])/(t[rb]-t[lb]))*DF[rb] #interpolation formula
    print('own interpol is',PtTi)
    print('diff interpol and own formula:', np.max(np.abs(PtTi-interpol)))
    Value = N*(     PtTi[-1]-PtTi[0] )
    PtTi[0] = 0                         #we want to sum over all the ZCB except the first one
    Value = Value + N*tau*K*0.01*np.sum(PtTi)
    return Value,PtTi

def n_sum(n):       #needed for spine points
    sum = 0
    for num in range(0,n+1):
        sum=sum+num
    return sum

def SpinePoints(t,FR,tau=1): #t: number of years of payments of the instruments, FR is fixed rates in %, tau time between payments 
    FR = FR*0.01 #rates are in percentages
    DF = np.zeros(len(FR))
    DF[0]=1
    PtTi = np.zeros(t[-1]+1)        # we make all the ZCB's for every year, until last year given
    for i in range(1,len(FR)):
        t_dif = int(t[i]-t[i-1])
        DF[i]= (1/   (FR[i] + 1 + FR[i]*n_sum(t_dif-1)/t_dif))*(1-FR[i]*(n_sum(t_dif-1)/t_dif*DF[i-1] + np.sum(PtTi)))       #sum of PtTi is all the PtTi's bef
        PtTi[t[i]] = DF[i]
        #count = 1
        for j in range(t[i-1]+1,t[i]):  #now we can make all the ZCB between the spine points
            #PtTi[j] = DF[i-1]*(count/n_sum(t_dif))+DF[i]*((t_dif-count)/n_sum(t_dif))
            #count=count+1
            PtTi[j] = (1- (j-t[i-1])/(t_dif))*DF[i-1]+((j-t[i-1])/(t_dif))*DF[i]  
            
    #print('from spine points:', PtTi)
    return DF
    
    
    

def Maincalculation():
    N = 1.0
    #fixed rates instruments denoted in percentage
    K1 = 5.
    K2 = 6.
    K3 = 7.
    K5 = 8.
    K10 = 12.
    PtT0 = 1.
    
    #annual payment dates
    tau1 = 1
    tau2 =  0.5
    tau3 = 0.4
    T0 = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T5 = 5
    T10 = 10
    
    PtT1 = 1/(K1*0.01+1)
    PtT2 = (1/(K2*0.01+1))*(1-K2*0.01*PtT1)
    PtT3 = (1/(K3*0.01+1))*(1-K3*0.01*(PtT1+PtT2))
    PtT5 = (1/(K5*0.01*1.5+1))*(1-K5*0.01*(PtT1+PtT2 + 1.5*PtT3))
    PtT10 = (1/(3*K10*0.01+1))*(1-K10*0.01*(PtT1+PtT2+1.5*PtT3+3.5*PtT5))
    
    t = np.array([0,1,2,3,5,10])      #T values of the spine points
    FR = np.array([0,K1,K2,K3,K5,K10])
    DF = [PtT0,PtT1,PtT2,PtT3,PtT5,PtT10]
    
    Swap1 = N*(PtT1-PtT0 + tau1*K1*0.01*PtT1) #with only payment at y1
    Swap2 =  N*(PtT2-PtT0 + tau1*K2*0.01*(PtT1+PtT2))        #with payments at y1 and y2
    Swap3 =  N*(PtT3-PtT0 + tau1*K3*0.01*(PtT1+PtT2+PtT3))   # with payments at y1,y2,y3
    
    print('Discount Factors are ',DF)
    
#    print('Swap1 value is ',Swap1)
#    print('Swap2 value is ',Swap2)
#    print('Swap3 value is ',Swap3)
#    
#    #swap with semi-annual payments: 
    PtTi = P(tau2, DF,T10) #array of PtTi's 
#    print('PtTi are', PtTi)
    Swap_SA_3y = Swap(T0,T3,N,K3,tau2,PtTi)        #zou
#    Swap_SA_10y = Swap(T0,T10,N,K10,tau2,PtTi)
#    Swap1_Test = Swap(T0,T1,N,K1,tau1,PtTi)
#    Swap2_Test = Swap(T0,T2,N,K2,tau1,PtTi)
#    Swap3_Test = Swap(T0,T3,N,K3,tau1,PtTi)
#    print('test annual swap1 value', Swap1_Test)
#    print('test annual swap2 value', Swap2_Test)
#    print('test annual swap3 value', Swap3_Test)
    
    print('semi annual swap_3y value', Swap_SA_3y)
    print('diff semi annual swap should have with swap3:',-0.25*PtTi[6]+0.25*PtTi[0])           #should be difference with swap 3
#    print('semi annual swap_10y value', Swap_SA_10y)
    T = np.linspace(T0,T10,21)
    interpol = np.interp(T,[0,1,2,3,5,10],[PtT0,PtT1,PtT2,PtT3,PtT5,PtT10])
    print('diff interpolation and P() is', np.max(PtTi-interpol))
    
#    newSwap1,P1 = NewSwap(T0,T1,N,K1,tau1,DF,t)
#    newSwap2,P2 = NewSwap(T0,T2,N,K2,tau1,DF,t)
#    newSwap3,P3 = NewSwap(T0,T3,N,K3,tau1,DF,t)
#    print('test new annual swap1 value', newSwap1)
#    print('test new annual swap2 value',newSwap2)
#    print('test new annual swap3 value',newSwap3)
    
    newSwap_SA_3y,P4 = NewSwap(T0,T3,N,K3,tau2,DF,t)
    print('test new semi annual swap 3y value',newSwap_SA_3y)
    newSwap_SA_2y,P5 = NewSwap(T0,T2,N,K2,tau3,DF,t)
    print('test new semi annual swap 3y value',newSwap_SA_2y)       #swap with tau = 0.4
    
    SP = SpinePoints(t,FR,tau=1)
    print(SP)
    print('maximum absolute diff between DF and Spine points from function:', np.max(np.abs(SP-DF)))

    
    


#    x = np.linspace(0,T10,21)
#    plt.plot(x,P(tau2,DF,T10))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
Maincalculation()