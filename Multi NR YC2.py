# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:39:07 2019

@author: jonkm
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import math
import pdb
import timeit

#blabla 271-0
#weer iets toegevoegd
#kijken of het lukt
def PriceIRS(Ta,Tb,N,K,tau,DF,t):    # t values at which we have the Discount Factors
                                        # we use linear interpolation!

    Num_payments = int((Tb-Ta)/tau)     #number of payments
                                        #Ta and Tb are the first and last payment dates in years
                                        #PtTi are the discount factors not per year, per payment date
    T = np.linspace(Ta,Tb,Num_payments+1)   #linspace at which to evaluate PtTi 
    PtTi = np.zeros(len(T))             #length of PtTi vector is eqeal to T
    for i in range(1,Num_payments+1):   # for every Ti we will find left and right bounds for interpolation
        for j in range(0,len(t)):       #search through the t values of the spine points
            if T[i] <= t[j]:            #for the first tj Ti is smaller: this is index left bound, j+1 is then right bound
                if j==0:                # if j==0, left bound is P(t,T0) = 1
                    PtTi[i] = (1-(T[i] - 0)/(1-0))*1 + ((T[i]-0)/(1-0))*DF[j] #interpolation formula
                    break
                else:
                    lb = j-1
                    rb = j
                    break                  #breaks out of second loop, the right j is found  
        if j!=0:
            a = t[lb]
            b = T[i]
            c =t[rb]
            d=DF[lb]
            e = DF[rb]
            PtTi[i] = (1-(T[i]-t[lb])/(t[rb]-t[lb]))*DF[lb] + ((T[i]-t[lb])/(t[rb]-t[lb]))*DF[rb] #interpolation formula
    Value = N*(     PtTi[-1]- 1 )  #P(t,T0) = 1
                          #we want to sum over all the ZCB except the first one
    Value = Value + N*tau*K*0.01*np.sum(PtTi) #in this sum PtTi[0] = 0
    PtTi[0] = 1
    return Value,PtTi

def Bootstrap(FR,t,q,N,tau,df_in):
    df_0=df_in
    for i in range(0,len(t)):           #loop through calibration swaps
        FR_i = FR[0:i+1]                #makes array of elements with index 0 up to and including i-1 (not i)
        t_i = t[0:i+1]
        q_i = q[0:i+1]
        tau_i = tau[0:i+1]
        
        df_1 = MultiNR(FR_i,t_i,df_0,q_i,N,tau_i,tolerance = 10**(-15),maxIter = 100)     #gives array of spine discount factors for i swaps
        #print('df1',df_1)
        if i!= len(t)-1:                            #for the last element we dont need to make new df vector
            df_0 = np.zeros(len(df_1)+1)            #make new set of DF's, one longer
            for j in range(-1,len(df_1)):            #loop through all elements but last one and set equal to previous
                df_0[j]=df_1[j]                     #loop starts at -1: hence last element of df0 is same as previous
        #print('df_in',df_0)
    return df_1
        
        
        

def JacoSolver(FR,t,DF,tau_array,N): #DF array of discount factors, t maturity times of swaps i.e. times of DF and array of fixed rates
    FR = 0.1*FR           #fixed rates are given in percentages
    h = 10**(-5)        #for finite difference scheme
    J = np.zeros((len(t),len(t)))       #Jacobian matrix 
    for i in range(0,len(t)):           #loop through spine swaps 
        for j in range(0,len(t)):       #loop through spine DF
            if j>i: 
                J[i,j] = 0              #derivative is zero for index DF > index Value
            else: 
                dDF = np.zeros(len(DF))         #array with only 0.5*h on position of DF[j]
                dDF[j] = 0.5*h                  #central difference f(x+0.5h)-f(x-0.5h)
                Ta = 0                          #Ta is current time
                Tb = t[i]                       #maturity of swap is time position of DF
                V1,PtTi1 = PriceIRS(Ta,Tb,N,FR[i],tau_array[i],DF+dDF[j],t)
                V2,PtTi2 = PriceIRS(Ta,Tb,N,FR[i],tau_array[i],DF-dDF[j],t) 
                J[i,j] = (V1 - V2)/h                            #central difference
    return J

def MultiNR(FR,t,DF,q,N,tau,tolerance = 10**(-15),maxIter = 100): #DF initual guess, q market quotes 
    n = 1
    h=10**(-5)
    df_0 = DF                   #initual guess is DF
    if len(t)==1:
        while n<maxIter:
            V,PtTi = PriceIRS(0,t[0],N,FR[0],tau[0],df_0,t)
            pv = V-q[0]
            dif1 = df_0+0.5*h
            dif2 = df_0-0.5*h
            V1,PtTi1 = PriceIRS(0,t[0],N,FR[0],tau[0],dif1,t)
            V2,PtTi2 = PriceIRS(0,t[0],N,FR[0],tau[0],dif2,t)
            pv_diff = (V1-V2   )/h
            df_1 = df_0 - pv/pv_diff
            if abs(df_1-df_0) < tolerance :
                break
            df_0 = df_1
            n=n+1
        if n==maxIter:
          print('maximum iteration is reached, no solution')  
        return df_1            
    else:
        while n < maxIter:           #continue until solution is found or maxIter has reached
            pv = np.zeros(len(t)) #arry with present values of spine points, t0 not included                                                  #if n==1 calculate V, otherwise at the end
            for i in range(0,len(t)):
                V, PtTi = PriceIRS(0,t[i],N,FR[i],tau[i],df_0,t)        
                pv[i] = V-q[i]                                 #pv present value minus market quote
            Jinv = np.linalg.inv(JacoSolver(FR,t,df_0,tau,N))
            df_1 = df_0 - np.matmul(Jinv,pv)                    #NR iteration formula
            if np.max(abs(df_1-df_0)) < tolerance: #if tolerance is reached
                break
            df_0 = df_1                     #prepare for new iteration
            n = n+1
           # print(n)
        if n == maxIter: 
            print('maximum iteration is reached, no solution')
    return df_1

       
def Maincalculation():
    N = 1.0
    #fixed rates instruments denoted in percentage
    #Ti are maturity times
    #tau_i is time between payment dates in years
    T0=0
    #calibration swaps: 
    #swap 1
    K1 = 5. 
    T1 = 1
    tau1 = 1
    #swap 2
    K2 = 6. 
    tau2 =  1.0
    T2 = 2
    #swap 3
    K3 = 7.
    tau3 = 1
    T3 = 3
    #swap 4
    K4 = 8
    tau4 = 1
    T4 = 5   
    #swap 5
    K5=12.
    tau5 = 1 
    T5 = 10
    
    
    PtT0 = 1.
    PtT1 = 1/(K1*0.01+1)
    PtT2 = (1/(K2*0.01+1))*(1-K2*0.01*PtT1)
    PtT3 = (1/(K3*0.01+1))*(1-K3*0.01*(PtT1+PtT2))
    PtT5 = (1/(K4*0.01*1.5+1))*(1-K4*0.01*(PtT1+PtT2 + 1.5*PtT3))
    PtT10 = (1/(3*K5*0.01+1))*(1-K5*0.01*(PtT1+PtT2+1.5*PtT3+3.5*PtT5))
    DF = [PtT1,PtT2,PtT3,PtT5,PtT10]
    print('Spine discount factors calculated by hand: ',DF)
    
    t = np.array([T1,T2,T3,T4,T5])            #T values of the spine points
    FR = np.array([K1,K2,K3,K4,K5])           # array of fixed rates of instruments
    tau_array = np.array([tau1,tau2,tau3,tau4,tau5]) #array of time between payment dates
    df_initual_guess = np.zeros(len(t))
    q_m = np.zeros(len(t))                      #array of market quotes, set to 0
#    
#    t1 = np.array([T1])            #T values of the spine points
#    FR1 = np.array([K1])           # array of fixed rates of instruments
#    tau_array1 = np.array([tau1]) #array of time between payment dates
##    df_initual_guess = np.array([0.2,0.6,0.8,0.9,0.9])
##    df_initual_guess = np.array([1,1,1,1,1])
#    df_initual_guess1 = np.array([1])
#    q_m1 = np.zeros(len(t1))
#   Spinepoints = MultiNR(FR1,t1,df_initual_guess1,q_m1,N,tau_array1,tolerance = 10**(-15),maxIter = 100) 

    
    for i in range(0,len(t)):
        V,PtTi = PriceIRS(T0,t[i],N,FR[i],tau_array[i],DF,t)
        print('price swap',i+1,' calculated with theoretical DF"s',V)
    
#    J = JacoSolver(FR,t,DF,tau,N)
#    Jinv = np.linalg.inv(J)
#    print('solution of Jacobian matrix',J)
#    print('J inverse is: ',Jinv)
    
    Spinepoints = MultiNR(FR,t,df_initual_guess,q_m,N,tau_array,tolerance = 10**(-15),maxIter = 100)
    print('Spine Discount factor with NR, ', Spinepoints)
    print('max abs diff is', np.max(abs(DF[0:len(Spinepoints)]-Spinepoints)))
#    df_in = np.ones(1)
    SP_bootstrap = Bootstrap(FR,t,q_m,N,tau_array,df_in=np.ones(1))
    print('SP with bootstrap:',SP_bootstrap)
    print('max abs diff bootstrap with DF is', np.max(abs(DF-Spinepoints)))
    
    #----------------------excel test
    t_excel = np.array([0.5,1,1.5,2,3,5,7.5,9.5])
    FR_excel = np.array([1.5,2.5,3,5.5,7,10.5,11,12.5])
    tau_array_excel = np.ones(len(t_excel))*0.5
    q_m_excel = np.zeros(len(t_excel))
    df_initual_guess_excel1 = np.zeros(len(t_excel))
    df_initual_guess_excel2 = np.ones(len(t_excel))*0.5
    df_initual_guess_excel3 = np.ones(len(t_excel))
    print('excel test-----------------------------------------')
    
    Spinepoints = MultiNR(FR_excel,t_excel,df_initual_guess_excel1,q_m_excel,N,tau_array_excel,tolerance = 10**(-15),maxIter = 100)
    print(Spinepoints)
    
    n = 1
    timeBS = 0
    while n<100:
        start = timeit.timeit()
        SP_bootstrap_e = Bootstrap(FR_excel,t_excel,q_m_excel,N,tau_array_excel,df_in=np.ones(1))
        end = timeit.timeit()
        n = n+1
        timeBS = timeBS + end - start
    timeBS = timeBS/100
    
    n = 1
    timeNR = 0
    while n<100:
        start = timeit.timeit()
        Spinepoints = MultiNR(FR_excel,t_excel,df_initual_guess_excel3,q_m_excel,N,tau_array_excel,tolerance = 10**(-15),maxIter = 100)
        end = timeit.timeit()
        n = n+1
        timeNR = timeNR + end - start
    timeNR = timeNR/100
    
    print('Spine discount factors with bootstrap NR, calibrated on test excel swaps',SP_bootstrap_e)
    print('elapsed time for bootstrap NR is',timeBS)
    print('elapsed time for One-Time NR is',timeNR)
#    for i in range(0,len(t_excel)):
#        V,PtTi = PriceIRS(T0,t_excel[i],N,FR_excel[i],tau_array_excel[i],SP_bootstrap_e,t_excel)
#        print('price swap',i+1,' from excel calculated with DFs from bootstrap:',V)
    

    

Maincalculation()