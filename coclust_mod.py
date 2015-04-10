#!/usr/bin/env python
# -*- coding: utf-8 -*-

corpus = "classic3"

from pyspark import SparkContext
from pyspark import SparkConf
from scipy.io import loadmat
import itertools
import numpy as np
import scipy.sparse as sps

from sklearn.metrics.cluster import normalized_mutual_info_score

#TODO:remove
np.random.seed(0)

conf = SparkConf()
conf.set("spark.executor.memory", "6g")
conf.set("spark.cores.max", "2")
conf.setAppName("tests")
sc = SparkContext("local", conf=conf)

###
# Paramètres
###

if corpus == "classic3":
    mat = loadmat("../tests/classic3.mat")
    A_orig = mat['A'] # nd_array
    A_orig = A_orig.todense()
    labels = mat['labels']
    K = 3 # nombre de clusters

elif corpus == "cstr":
    mat = loadmat("../tests/cstr.mat")
    A_orig = mat['fea'] # nd_array
    A_orig = A_orig
    labels = mat['gnd']
    K = 4 # nombre de clusters

###
# Prétraitement
###

nb_rows = A_orig.shape[0]
nb_cols = A_orig.shape[1]

# labels
labels = labels.tolist()
labels = list(itertools.chain.from_iterable(labels))
label = []
for l in labels:
    label.append(int(l-1))
labels = label

#TODO: partir d'un fichier (ou bdd)
# 
# A_rrd est un RDD qui représente la matrice des poids A :
# Chaque élément du RDD est un couple (i, l) où
# - i est le numéro de la ligne
# - l est une matrice sparse représentant la ligne i
A_rdd = sc.parallelize(((i, sps.csc_matrix(A_orig[i, :])) for i in range(nb_rows)))
A_rdd.persist()

###
# Initialisation aléatoire
###

# Z matrice sparse
# Zij = 1 <=> ligne i dans le cluster j
Z_array = np.random.randint(K, size=nb_rows)
Z = np.zeros((nb_rows, K))
Z[np.arange(nb_rows) , Z_array] = 1
Z = sps.lil_matrix(Z, dtype=float)

# W matrice sparse
# Wij = 1 <=> colonne i dans le cluster j
W_array = np.random.randint(K, size=nb_cols)
W = np.zeros((nb_cols, K))
W[np.arange(nb_cols) , W_array] = 1
W = sps.lil_matrix(W, dtype=float)

###
# Fonctions utiles
###

# pour un couple (i, ligne) où
# - i est un entier : numéro de la ligne
# - ligne est une matrice sparse
# retourne (max_index, i) où
# max_index est le max de la ligne i
def assign_rows(arg):
    (i, l) = arg
    row_vector = l.getrow(0)
    max_index = row_vector.indices[row_vector.data.argmax()] if row_vector.nnz else 0
    return (max_index, i)

# transforme une matrice RDD en matrice scipy
# - n est le nombre de lignes
# - m est le nombre de colonnes
def rdd_to_matrix(rdd, n, m):
    return (sps.vstack(rdd.sortByKey().values().collect()))

###
# Algorithme
###

row_sums_rdd = A_rdd.map(lambda (i, l): (i, l.sum()))
col_sums = A_rdd.reduce(lambda (i, l), (i2, l2): (0, l+l2))[1]
N = float(col_sums.sum())

A_rdd.first()
print "DEBUG begin algo"

# calcul de B
minus_col_sums_N_b = sc.broadcast(-col_sums/N)
minus_indep = row_sums_rdd.map(lambda (i, l): (i, l * minus_col_sums_N_b.value))
B_rdd = A_rdd.join(minus_indep).map(lambda (i, (l1, l2)): (i, l1+l2))
B_rdd.persist()
#A_rdd.unpersist()

B_rdd.first()
print "DEBUG B_rdd"

# calcul de Bt
minus_row_sums_N_t = sps.csc_matrix([-e/N for (_, e) in row_sums_rdd.collect()])
minus_row_sums_N_t_b = sc.broadcast(minus_row_sums_N_t)
col_sums_t_rdd = sc.parallelize(zip(range(nb_cols), col_sums.toarray().flatten()))
minus_indep_t = col_sums_t_rdd.map(lambda (i, l): (i, l * minus_row_sums_N_t_b.value))
# A_t_rdd est la transposée de A sous la même forme que A_rdd
At_rdd = A_rdd.flatMap(lambda (i, l): [(j, (i, e)) for (e, j) in zip(l.getrow(0).data, l.getrow(0).indices)])
def f(data):
    (indexes, elements) = zip(*data)
    elements = np.array(elements)
    indexes = np.array(indexes)
    
    return sps.csc_matrix(((elements), (np.zeros(elements.size, dtype=np.int), indexes)), shape=(1, nb_rows))

At_rdd = At_rdd.groupByKey().map(lambda (j, x): (j, f(x.data)))
Bt_rdd = At_rdd.join(minus_indep_t).map(lambda (i, (l1, l2)): (i, l1+l2))
Bt_rdd.persist()

Bt_rdd.first()
print "DEBUG Bt_rdd"

m_begin = float("-inf")
change = True
while (change):
    change = False
    
    ###
    # Assignation des lignes dans les clusters
    ###
    
    # B * W
    W_broadcast = sc.broadcast(W)
    BW_rdd = B_rdd.map(lambda (i, l): (i, l.dot(W_broadcast.value)))
    
    # RDD de couple (max_index, i) où
    # max_index est le max de la ligne i de A * W * Dc
    # max_index est le cluster dans lequel on met la ligne i
    row_assignments = BW_rdd.map(assign_rows)
    
    # Construction de Z
    max_indexes, line_numbers = zip(*row_assignments.collect())
    max_indexes_array = np.array(max_indexes)
    n = max_indexes_array.size
    Z = sps.csc_matrix((np.ones(n), (np.array(line_numbers), max_indexes_array)), shape=(nb_rows, K))
    
    ###
    # Assignation des colonnes dans les clusters
    ###
    
    # Bt * Z
    Z_broadcast = sc.broadcast(Z)
    BtZ_rdd = Bt_rdd.map(lambda (i, l): (i, l.dot(Z_broadcast.value)))
    
    # RDD de couple (max_index, j) où
    # max_index est le max de la ligne j de At * Z * Dr
    # max_index est le cluster dans lequel on met la colonne j
    col_assignments = BtZ_rdd.map(assign_rows)
    
    # Construction de W
    max_indexes, column_numbers = zip(*col_assignments.collect())
    max_indexes_array = np.array(max_indexes)
    n = max_indexes_array.size
    W = sps.csc_matrix((np.ones(n), (np.array(column_numbers), max_indexes_array)), shape=(nb_cols, K))
    
    ###
    # Critère
    ###
    
    #TODO: déplacer au début de boucle pour éviter de recalculer B * W
    # B * W
    W_broadcast = sc.broadcast(W)
    BW_rdd = B_rdd.map(lambda (i, l): (i, l.dot(W_broadcast.value)))
    print "DEBUG - BW_rdd"
    BW = rdd_to_matrix(BW_rdd, nb_rows, K)
    print "DEBUG - BW"
    
    # Zt * B * W
    k_times_k= (Z.T) * BW
    
    m_end = np.trace(k_times_k.todense())# pas de trace pour sp ...
    print "*** m at the end of the loop *** %f" % m_end

    if np.abs(m_end - m_begin) > 1e-9 :
        m_begin = m_end
        change = True
    
    print "Criterion %f" % m_end
    
#TODO: utilité de sparse?


# evaluation
predicted = np.argmax(Z.toarray(), axis=1)
print "NMI" , normalized_mutual_info_score(labels, predicted)

