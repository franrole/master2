import sys
import getopt
import re
import os,glob
from math import * 
import numpy as np
from numpy import *
from collections import *

import scipy.sparse as sp
import marshal
import cPickle
import pickle


from random import randint
import itertools
##from scipy.io import loadmat, savemat
##from sklearn.datasets import fetch_20newsgroups
##from sklearn.feature_extraction.text import  CountVectorizer
##from sklearn.preprocessing import binarize
##from sklearn.feature_extraction.text import TfidfTransformer

##if __name__ == "__main__":
def essai() :
        #A=sp.lil_matrix(np.arange(12).reshape(4,3) ,dtype=float)
        print "essai"
        a = np.arange(5)
        print a

#corpus="cstr"

##if corpus == "cstr" :
##        
##        K=4
##
##        mat=loadmat("../data/cstr.mat")
##
##        labels=mat['gnd']
##        labels=labels.tolist()
##        labels = list(itertools.chain.from_iterable(labels))
##        label=[]
##        for l in labels:
##                label.append(int(l-1))
##        labels=label
##
##        A_orig = mat['fea'] # nd_array
##        A=sp.lil_matrix(A_orig,dtype=float)
##
##
##
##        nb_rows=A.shape[0]
##        nb_cols=A.shape[1]
##
##
##        W_a=np.random.randint(K,size=nb_cols)
##        W=np.zeros((nb_cols,K))
##        W[np.arange(nb_cols) , W_a]=1
##        W=sp.lil_matrix(W,dtype=float)
##
##
##        row_sums=sp.lil_matrix(A.sum(axis=1))
##        col_sums=sp.lil_matrix(A.sum(axis=0))
##        N=float(A.sum())
##
##
##
##        # Loop
##
##        rcut_begin=  1e9 #float("-inf") 
##
##        change=True
##        while (change):
##            change=False
##
##            # Reassign rows    
##            AW=A * W
##            col_clust_cards= W.sum(axis=0) # cards =matrix([[3],[12],...]
##            cards=np.diag( (1.0 / np.array(col_clust_cards).flatten() ))
##            AW_arr=AW.toarray()
##            AW_arr= AW_arr.dot(cards)
##            for idx , k  in enumerate(np.argmax(AW_arr, axis=1)) :
##                Z[idx,:]=0
##                Z[idx,k]=1
##
##            Z_arr=Z.toarray()
##            W_arr=W.toarray()
##            #k_times_k = D_r.dot(Z_arr.T).dot(A_orig).dot(W_arr).dot(D_c)
##            k_times_k = (Z_arr.T).dot(A_orig).dot(W_arr).dot(D_c)
##            criterion=np.trace(k_times_k)
##            
##
##            # Reassign columns
##            AtZ=(A.T) * Z
##            row_clust_cards= Z.sum(axis=0) # cards =matrix([[3],[12],...]
##            cards=np.diag(  (1.0 / np.array(row_clust_cards).flatten() )  )
##            AtZ_arr=AtZ.toarray()
##            AtZ_arr=AtZ_arr.dot(cards)
##            for idx , k  in enumerate(np.argmax(AtZ_arr, axis=1)) :
##                W[idx,:]=0
##                W[idx,k]=1
##
##            row_clust_cards= Z.sum(axis=0)
##            D_r=np.diag(  (1.0 / np.array(row_clust_cards).flatten() )  )
##            col_clust_cards= W.sum(axis=0) 
##            D_c=np.diag( (1.0 / np.array(col_clust_cards).flatten() ))
##
##            k_times_k= (Z.T) * AW
##            trace=np.trace(k_times_k.todense())
##            rcut_end= N-trace
##            print "       cut at the end of the loop %f (criterion=%f)" % (rcut_end , criterion)
##
##            if (rcut_begin - rcut_end) > 1e-9 :
##                rcut_begin=rcut_end
##                change=True
##
##        ##print
##        ##print "Result" 
##        ##print Z.todense()
##        ##print
##
##
##        predicted=np.argmax(Z.toarray(), axis=1)
##
####print labels
####print
####print predicted
##
##from sklearn.metrics.cluster import normalized_mutual_info_score
##print "NMI" , normalized_mutual_info_score(labels, predicted)
##
##
##
##    
##
##    
##    
##    
##
##
##
##
##
##
