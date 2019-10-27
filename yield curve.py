# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:04:44 2019

@author: jonkm
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import math


def Theta(ita,lbda,t):#theta function
    return 0.06*t+0.15+0.06/lbda + (ita**(2)/(2*lbda**2))*(1-np.exp(-2*lbda*t))  
#    
def B(lbda, t,T): #B_r function
    tau = T-t
    return 1/lbda*(np.exp(-lbda*tau)-1)
#
def A(eta,lbda,t,T,acc):#A_r function
    tau = T-t
    steps = np.linspace(t,T,acc)  #making the steps for calculating the integral part
    dt = tau/(acc*1.0-1)
    #C is A - the integral part
    C= eta**(2)/(4*lbda**(3))*(np.exp(-2*lbda*tau)*(4*np.exp(lbda*tau)-1)-3)+(eta**(2)*tau)/(2*lbda**2)
    beta = [0]*len(steps) #this will be the integral part
    beta[0] = 0 #initial value
    for i in range(0,len(steps)-1): #we will use the trapezium rule
        beta[i+1] = beta[i]+0.5*dt*(Theta(eta, lbda,steps[i])*B(lbda,steps[i],T) + Theta(eta, lbda,steps[i+1])*B(lbda,steps[i+1],T))
    
    return lbda*beta[-1]+C #A=lbda*integralPart+C

def P(t0,T,NoOfSteps,y,r0,eta,lbda,acc):
    P = np.exp(A(eta,lbda,t0,T,acc)+B(lbda,t0,T)*r0)
    return P
    

def priceIRSwap(K,T,p): #input is strike K, strip of discount factors at Ti and T array of payment dates Ti
    Ptk = 0                 #discount factors p = P(t0,Ti)
    #T array varies, p size is always 3
    for i in range(0,len(T)-1): #T is array input
        Ptk = Ptk + p[i]
    Vt0 = 1 - p[len(T)-2] - K*Ptk #V = P(t,Tm) + P(t,Tn) - K*P(t,Tk), P(t,Tm) = P(t,T0) = P(0.0) = 1
    return Vt0                  

def Jaco(K,T,p): #input is array with strike prices, array with maturity times/payment dates, p array of discount factors
    dp = 10**(-5)
    T1 = np.linspace(0,T[0],2)  #array of payments for swap 1
    T2 = np.linspace(0,T[1],3)              #array of payments for swap 2
    T3 = np.linspace(0,T[2],4)
    Ti = np.array([T1,T2,T3])
    J = np.zeros((3,3))
        
    for i in range(0,3): #row
        for j in range(0,3): #column 
            dpV = np.zeros(len(p))               #vector with only dp on position of pi
            dpV[j] = 0.5*dp                     #on position p[j] we want to at dp
            J[i,j] = (priceIRSwap(K[i],Ti[i],p+dpV) - priceIRSwap(K[i],Ti[i],p-dpV))/dp  #central difference scheme
    return J 

def MultiNR(K,T,pi,q,error = 10**(-4),maxIter = 20): #pi is initual guess, error is tolerance
    dp=10**(-5) #h-->0
    T1 = np.linspace(0,T[0],2)  #array of payments for swap 1
    T2 = np.linspace(0,T[1],3)              #array of payments for swap 2
    T3 = np.linspace(0,T[2],4)
    Ti = np.array([T1,T2,T3])
    
    increment = 0.01 #for searchloop
    #pi is initual guess
    y = np.zeros(len(pi)); #this is going to be the output
    while np.min(pi) <= 20: #discount factors wont be 10, stop searching for initual guess
        p0 = pi
        n=1 
        while n<= maxIter:
            pv = np.array([priceIRSwap(K[0],Ti[0],p0),priceIRSwap(K[1],Ti[1],p0),priceIRSwap(K[2],Ti[2],p0)])-q
            Jinv = np.linalg.inv(Jaco(K,T,p0))
            p1 = p0 - np.matmul(Jinv, pv)
            if abs(np.min(p1)-np.min(p0)) <error and p0[0]!=p1[0] and p0[1]!=p1[1] and p0[2]!=p1[2]:
                break #goes outside while loop
            p0 = p1
            n = n+1
        if abs(np.min(p1)-np.min(p0)) < error and p0[0]!=p1[0] and p0[1]!=p1[1] and p0[2]!=p1[2]: #if error is small and they are not the same, we have found solution
            break           #found solution, go outside first while loop
        #if not the case, we are still in while loop, try new initual guess:
        pi = pi+increment
        if np.min(pi) >= 10: #pi wont get zo big
            pi = np.zeros(len(p0))
            increment = increment/2         #try new initial search for good initual guess, now with smaller increments for searching
    df = p1
    return df
        
def Maincalculation():
    lbda = 0.5
    eta = 0.03
    acc = 40 #accuracy of integral of Ar
    t0 = 0

    NoOfSteps = 100
    T0 = 0 #Tm = T0
    T1 = 1
    T2 = 2
    T3 = 3
    
    y=1 #one year
    
    T = np.linspace(T1,T3,3)
    #fixed rates, swaps V(t0) equal zero 
    K1= 0.01
    K2 = 0.0214
    K3 = 0.038
    K = np.array([K1,K2,K3])
    
    qm = np.zeros(3) #market prices are zero at t0
    
    #pi = np.array([0.8,0.6,0.4]) #initual guess for set of discount factors , 
    pi = np.array([0,0,0])
    V0 = priceIRSwap(K[0],T,pi)
    print('example swap', V0)
    J = Jaco(K,T,pi)
    print('solution of Jacobian matrix',J)
    print(np.linalg.inv(J))

    spinePoints = MultiNR(K,T,pi,qm,error = 10**(-5),maxIter = 20)
    print('optimal spine points:',spinePoints)
    
    print('swap prices',priceIRSwap(K,T,spinePoints))
    
    
    
Maincalculation()