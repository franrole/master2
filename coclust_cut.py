#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark import SparkConf
from scipy.io import loadmat
import itertools
import numpy as np
import scipy.sparse as sps

from sklearn.metrics.cluster import normalized_mutual_info_score

conf = SparkConf()
conf.set("spark.executor.memory", "6g")
conf.set("spark.cores.max", "2")
conf.setAppName("tests")
sc = SparkContext("local", conf=conf)
###
# Prétraitement
###

# la matrice des poids
mat = loadmat("../tests/cstr.mat")

A_orig = mat['fea'] # nd_array
A_orig = A_orig#.todense()

nb_rows = A_orig.shape[0]
nb_cols = A_orig.shape[1]

# labels
labels=mat['gnd']
labels=labels.tolist()
labels = list(itertools.chain.from_iterable(labels))
label=[]
for l in labels:
        label.append(int(l-1))
labels=label

#TODO: partir d'un fichier (ou bdd)
# 
# A_rrd est un RDD qui représente la matrice des poids A :
# Chaque élément du RDD est un couple (i, l) où
# - i est le numéro de la ligne
# - l est une matrice sparse représentant la ligne i
A_rdd = sc.parallelize(((i, sps.csc_matrix(A_orig[i, :])) for i in range(nb_rows)))
A_rdd.persist()

print "DEBUG - A_rdd"
A_rdd.first()
print "DEBUG - END - A_rdd"


# A_t_rdd est la transposée de A sous la même forme que A_rdd
At_rdd = A_rdd.flatMap(lambda (i, l): [(j, (i, e)) for (e, j) in zip(l.getrow(0).data, l.getrow(0).indices)])
def f(data):
    (indexes, elements) = zip(*data)
    elements = np.array(elements)
    indexes = np.array(indexes)
    
    return sps.csc_matrix(((elements), (np.zeros(elements.size, dtype=np.int), indexes)), shape=(1, nb_rows))

At_rdd = At_rdd.groupByKey().map(lambda (j, x): (j, f(x.data)))

#At = A_orig.T
#At_rdd = sc.parallelize(((i, sps.csc_matrix(At[i, :])) for i in range(nb_cols)))
At_rdd.persist()

print "DEBUG - At_rdd"
At_rdd.first()
print "DEBUG - END - At_rdd"

###
# Paramètres
###

# nombre de clusters
K = 4

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
    def f(arg):
        (i, l) = arg
        row_vector = l.getrow(0)
        col_indices = row_vector.indices
        row_indices = i * np.ones(col_indices.size, dtype="int")
        return (row_indices, col_indices, row_vector.data)
    
    l = rdd.map(f).collect()
    
    row_indices = np.array([])
    col_indices = np.array([])
    data = np.array([])
    for (r_i, c_i, d) in l:
        row_indices = np.append(row_indices, r_i)
        col_indices = np.append(col_indices, c_i)
        data = np.append(data, d)
        
    return sps.csc_matrix((data, (row_indices, col_indices)), shape=(n, m))

###
# Algorithme
###

change = True
#TODO critère d'arrêt
for k in range(5):
    change = False
    
    ###
    # Assignation des lignes dans les clusters
    ###

    # A * W
    W_broadcast = sc.broadcast(W)
    AW_rdd = A_rdd.map(lambda (i, l): (i, l.dot(W_broadcast.value)))
    
    print "DEBUG - AW_rdd"
    AW_rdd.first()
    print "DEBUG - END - AW_rdd"
    
    # Nombre de colonnes par cluster
    col_clust_cards = np.array(W.sum(axis=0)).flatten()
    
    # Dc
    Dc = np.array((1.0 / col_clust_cards))
    
    # A * W * Dc
    Dc_broadcast = sc.broadcast(Dc)
    AWDc_rdd = AW_rdd.map(lambda (i, l): (i, sps.csc_matrix(l.multiply(Dc_broadcast.value))))
    
    print "DEBUG - AWDc_rdd"
    AWDc_rdd.first()
    print "DEBUG - END - AWDc_rdd"
    
    # RDD de couple (max_index, i) où
    # max_index est le max de la ligne i de A * W * Dc
    # max_index est le cluster dans lequel on met la ligne i
    row_assignments = AWDc_rdd.map(assign_rows)
    
    print "DEBUG row_assignments"
    row_assignments.first()
    print "END row_assignments"
    
    # Construction de Z
    max_indexes, line_numbers = zip(*row_assignments.collect())
    max_indexes_array = np.array(max_indexes)
    n = max_indexes_array.size
    Z = sps.csc_matrix((np.ones(n), (np.array(line_numbers), max_indexes_array)), shape=(nb_rows, K))
    
    ###
    # Assignation des colonnes dans les clusters
    ###
    
    # At * Z
    Z_broadcast = sc.broadcast(Z)
    AtZ_rdd = At_rdd.map(lambda (i, l): (i, l.dot(Z_broadcast.value)))
    
    print "DEBUG AtZ_rdd"
    AtZ_rdd.first()
    print "END AtZ_rdd"
    
    # Nombre de lignes par cluster
    row_clust_cards = np.array(Z.sum(axis=0)).flatten()
    
    # Dr
    Dr = np.array((1.0 / row_clust_cards))
    
    # At * Z * Dr
    Dr_broadcast = sc.broadcast(Dr)
    AtZDr_rdd = AtZ_rdd.map(lambda (i, l): (i, sps.csc_matrix(l.multiply(Dr_broadcast.value))))
    
    print "DEBUG AtZDr_rdd"
    AtZDr_rdd.first()
    print "END AtZDr_rdd"
    
    # RDD de couple (max_index, j) où
    # max_index est le max de la ligne j de At * Z * Dr
    # max_index est le cluster dans lequel on met la colonne j
    col_assignments = AtZ_rdd.map(assign_rows)
    
    print "DEBUG col_assignments"
    col_assignments.first()
    print "END col_assignments"
    
    # Construction de W
    max_indexes, column_numbers = zip(*col_assignments.collect())
    max_indexes_array = np.array(max_indexes)
    n = max_indexes_array.size
    W = sps.csc_matrix((np.ones(n), (np.array(column_numbers), max_indexes_array)), shape=(nb_cols, K))
    
    ###
    # Critère
    ###
    #TODO: placer au bon endroit
    
    AWDc = rdd_to_matrix(AWDc_rdd, nb_rows, K)

    Dr_diag = sps.dia_matrix(np.diag(Dr))
    
    trace = np.trace(Dr_diag.dot(Z.transpose()).dot(AWDc).todense())
    
    print "trace %f" % trace


# evaluation
predicted = np.argmax(Z.toarray(), axis=1)
print "NMI" , normalized_mutual_info_score(labels, predicted)