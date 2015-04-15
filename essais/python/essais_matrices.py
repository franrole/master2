#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""essais_matrices.py"""

# usage
# ./bin/spark-submit --master local[4] essais_matrices.py 

from pyspark import SparkContext
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import rand as sparse_rand

sc = SparkContext("local", "tests")

## Petites matrices
# matrice dense
t = np.array([[1, 2],
              [3, 4]])
p_t = sc.parallelize([t])       # la matrice sera copiée (quand on l'utilisera)
n_t = p_t.collect()[0]          # on récupère la matrice
print "n_t[1,1] %i" % n_t[1,1]  # doit afficher 4

# matrice sparse
# matrix([[4, 0, 9, 0],
#        [0, 7, 0, 0],
#        [0, 0, 0, 0],
#        [0, 0, 0, 5]])
row = np.array([0, 3, 1, 0])
col = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
m = coo_matrix((data, (row, col)), shape=(4, 4)) # construction de la matrice dans un format pour matrices sparse (COO)
p_m = sc.parallelize([m])
n_m = p_m.collect()[0]
e = n_m.getrow(1).todense()[0,1]  # terme [1,1]
print "n_m %i" % e                # doit afficher 7

## Matrices plus grandes
# matrice dense
m2 = np.random.randint(10**5, size=(10**5, 10))
print "taille de m2 : %i Mo" % (m2.nbytes/(1024**2)) # 7 Mo
p_m2 = sc.parallelize([m2])
n_m2 = p_m2.collect()[0]
print "n_m2 %i" % n_m2[10**4, 3]  # affiche un terme

# matrice sparse
m3 = sparse_rand(10**5, 10**2, density=0.1)
print "taille de m3 : %i Mo" % (m3.data.nbytes/(1024**2)) # 7 Mo
p_m3 = sc.parallelize([m3])
n_m3 = p_m3.collect()[0]
e = n_m3.getrow(1).todense()[0,3]  # terme [100,3]
print "n_m3 %i" % e

def multiplier(m, v):
    """multiplier une matrice m et un vecteur v
    
    m : rdd de vecteurs lignes
    v : vecteur
    """
    v_b = m.context.broadcast(v)
    return m.map(lambda l: l*v_b.value)

l1 = np.array([0, 3, 1, 0])
l2 = np.array([0, 3, 1, 2])
m = [l1, l2, l2]
v = np.array([1, 1, 1, 0])

rdd = sc.parallelize(m)
multiplier(rdd, v).collect()
