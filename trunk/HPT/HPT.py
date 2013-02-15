#This module is an implementation of the Hierarchical Performance Test
#proposed on the paper below
#Reference: [1]T. Chen, Y. Chen, Q. Guo, O. Temam, Y. Wu, and W. Hu,  
#"Statistical Performance Comparisons of Computers", 
#In Proceedings of HPCA-18, New Orleans, Louisiana, 2012.

import numpy as np
import scipy as sp
from scipy import stats
import math
import itertools
from ctypes import *
from memo import *

#Counts the number of sums of elements of all subsets of the integer sequence
#{1,...,n} that are no larger than an especified value
#input:
#   n: maximum value of the sequence
#   val: value used in comparisons
#output:
#   count: result
@memoized
def select(n, val):
    count = 0
    seq = range(1, n+1)
    for L in range(0, len(seq)+1):        
        for subset in itertools.combinations(seq, L):            
            if sum(subset) <= val:
                count += 1
    return count

#Counts the number of sums of elements of all subsets of the integer sequence
#{1,...,n} that are no larger than an especified value
#input:
#   n: maximum value of the sequenced
#   val: value used in comparisons
#output:
#   count: result
@memoized
def select2(n, size, val):
    count = 0
    seq = range(1, n+1)
    for subset in itertools.combinations(tuple(seq), size):            
        if sum(subset) <= val:
            count += 1
    return count

#Returns the critical value for the ranksum test
#input:
#   n1, n2: parameter of the critical value
#   alpha: significance level
#output:
#   critical value
def ranksumtable(n1, n2, alpha):    
    total = sp.misc.comb((n1+n2), n1)
    m = total * alpha
    for i in range(0, n1*(1+n1+2*n2)/2):
        if (select2((n1+n2),n1,i)) > m:
            return i-1

#Wilcoxon rank-sums test, adpated from scipy.stats.ranksums
#input:
#    x,y : array_like
#           The data from the two samples 
#output:
#    prob : float
#        p-value of the test
def ranksums(x, y):
    x,y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x,y))
    ranked = stats.rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x,axis=0)
    expected = n1*(n1+n2+1) / 2.0
    if len(x)>=12:
        z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)
        prob = 2 * stats.distributions.norm.sf(abs(z))        
    else:
        t = sp.misc.comb(len(x)+len(y), len(x))
        l = select2(len(x)+len(y), len(x), s - 1)        
        m = select2(len(x)+len(y), len(x), s) - l
        r = t - l -m
        if l < r:
            prob = float(l+m)/t        
        else: 
            prob = float(r+m)/t
        if prob > 0.5:
        	prob = 0.5
    return prob

#Wilcoxon signed-ranked test, adpated from scipy.stats.wilcoxon
#input:
#   x: array of differences
#output:
#   r_minus, r_plus = sums of ranks of negative and positive differences,
#                     respectively; used in comparisons
#   prob = the test's p-value
def wilcoxon_srt(x):
    d = x    
    d = np.array(d, dtype = float)
    count = len(d)	
    r = stats.rankdata(abs(d))    
    #################################
    r_zero = sum((d == 0)*r)/2.0  
    r_plus = sum((d > 0)*r) + r_zero
    r_minus = sum((d < 0)*r) + r_zero
    #################################
    #As in section 4.1 in [1]
    if count < 25:
        prob = float(select(count, r_minus))/(2**count)
    else:
        mn = count*(count+1.0)*0.25
        se = math.sqrt(count*(count+1)*(2*count+1.0)/24)
        z = (r_minus - mn)/se
        prob = stats.norm.cdf(z)        
    return r_minus, r_plus, prob

#Iplementation of Hierarchical Perfomance Test as proposed in [1]
#input:
#   A, B: measure matrices
#   speedup: desired speedup for reliability check. Default = 1.0
#   alpha_rs: confidence level for the ranksum tests. Default = 0.05
#   mode: if 0, bigger values are better; if 1, smaller values are better
#         Default = 1
#output:
#   p : HPT's p-value
#   r_minus, r_plus = sums of ranks of negative and positive differences,
#                     respectively; used in comparisons
def HPT(A, B, speedup = 1.0, alpha_rs = 0.05, mode = 1):
    if mode:
        A, B = B, A   
    D = []
    A = np.array(A, dtype = np.double)
    B = np.array(B, dtype = np.double)
    A /= np.double(speedup)
    la0 = len(A[0])
    la = len(A)
    if la0 <= 3:
        alpha_rs = 0.1
    for i in range(la):
        if la0 < 3:
            D.append(np.median(A[i]) - np.median(B[i]))
        elif la0 < 12:
            alldata = np.concatenate((A[i],B[i]))
            ranked = stats.rankdata(alldata)
            x = ranked[:la0]            
            s = np.sum(x,axis=0)
            lower = ranksumtable(la0, la0, alpha_rs)
            upper = la0*(la0+la0+1)-lower
            if s <= lower or s >= upper:
                D.append(np.median(A[i]) - np.median(B[i]))		
            else:
                D.append(0)                
        else:
            p = ranksums(A[i],B[i])
            if p <= alpha_rs:
                D.append(np.median(A[i]) - np.median(B[i]))
            else:
                D.append(0)
    r_minus, r_plus, p = wilcoxon_srt(D)
    return p, r_minus, r_plus

#Returns the maximum speedup that has an specified reliability
#input:
#   A, B: measure matrices
#   rel: desired reliability for maximum speedup check. Default = 0.95
#   alpha_rs: confidence level for the ranksum tests. Default = 0.05
#   acc: number of digits of output
#output:
#   su : maximum speedup calculated
#TODO: improve heuristics
def HPT_max_speedup(A, B, rel = 0.95, alpha_rs = 0.05, acc = 3, mode = 1): 
    su = 1
    delta = 0.1
    X = A
    Y = B    
    smaller = False
    p, rm, rp = HPT(X,Y, 1, alpha_rs, mode)
    if rp < rm:
        X, Y, rm, rp = Y, X, rp, rm        
        p = 1 - p
        smaller = True
    for i in range(acc):
        while p < 1 - rel and rp >= rm:
            su += delta
            p, rm, rp = HPT(X, Y, su, alpha_rs, mode)
        su -= delta
        delta *= 0.1
        p, rm, rp = HPT(X, Y, su, alpha_rs, mode)
    if smaller:
        return 1/su
    else:
        return su
