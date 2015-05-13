package coClustering

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{
  Vector,
  Vectors,
  Matrices,
  SparseMatrix,
  Matrix
}
import org.apache.spark.mllib.linalg.distributed.{
  MatrixEntry,
  CoordinateMatrix,
  BlockMatrix
}
import org.apache.spark.rdd.RDD

class ModularityWithBlockMatrix(
  var k: Int,
  maxIterations: Int = 30,
  blockSize: Int = 1024,
  epsilon: Double = 1e-9) {

  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  def run(cooMatrix: CoordinateMatrix): coClusteringModel = {
    val B = ModularityWithBlockMatrix.createBFromA(cooMatrix, blockSize)
    B.blocks.cache()

    val numCols = B.numCols().toInt
    val numRows = B.numRows().toInt
    
    val sc = B.blocks.sparkContext

    var W = ModularityWithBlockMatrix
      .createRandomW(sc, numCols, k, blockSize)

    var m_begin = Double.MinValue
    var change = true

    var Z: BlockMatrix = null
    var iter = 0

    while (change & iter <= maxIterations) {
      change = false

      var BW = B.multiply(W)

      Z = ModularityWithBlockMatrix
        .create_Z_or_W(BW, numRows, k, blockSize)

      var BtZ = B.transpose.multiply(Z)

      W = ModularityWithBlockMatrix
        .create_Z_or_W(BtZ, numCols, k, blockSize)

      var k_times_k = Z.transpose.multiply(BW)

      var m_end = ModularityWithBlockMatrix.trace(k_times_k)

      val diff = (m_end - m_begin)
      val diff_abs = if (diff < 0) -diff else diff
      if (diff_abs > epsilon) {
        m_begin = m_end
        change = true
      }
      println(iter)
      println("Criterion: " + m_end)
      iter = iter + 1
    }

    val res = new coClusteringModel()
    res.setRowLabelsFromBlockMatrix(Z)

    res

  }

}

object ModularityWithBlockMatrix {

  def train(
    sc: SparkContext,
    cooMatrixFilePath: String,
    nRows: Int,
    nCols: Int,
    k: Int,
    maxIterations: Int = 30,
    blockSize: Int = 1024,
    epsilon: Double = 1e-9): coClusteringModel = {

    val A_coo = cooMatrix(sc, cooMatrixFilePath, nRows, nCols)

    new ModularityWithBlockMatrix(k, maxIterations, blockSize, epsilon).run(A_coo)
  }

  /**
   * Créer une CoordinateMatrix à partir d'un fichier.
   *
   * @param sc       sparkcontext
   * @param filePath chemin vers le fichier contenant les valeurs de la matrice
   *                 où chaque ligne est de la forme i,j,e avec :
   *                 -  i index de la ligne
   *                 -  j index de la colonne
   *                 -  e élément de la ligne i et la colonne j
   * @param nRows    nombre de lignes de la matrice
   * @param nCols    nombre de la colonne de la matrice
   * @return         la matrice représentée par une CoordinateMatrix
   */
  def cooMatrix(sc: SparkContext, filePath: String, nRows: Int,
                nCols: Int): CoordinateMatrix = {
    val matrixEntries = sc.textFile(filePath).map { line =>
      val l = line.split(',')
      new MatrixEntry(l(0).toLong, l(1).toLong, l(2).toDouble)
    }

    new CoordinateMatrix(matrixEntries, nRows, nCols)
  }

  /**
   * Sommes des éléments de chaque colonne d'une matrice.
   *
   * @param cooMatrix la matrice sous forme de CoordinateMatrix
   * @return          un tableau des sommes des éléments de chaque colonne
   */
  def colSums(cooMatrix: CoordinateMatrix): Array[Double] = {
    cooMatrix.entries.aggregate(
      Array.fill[Double](cooMatrix.numCols().toInt)(0))(
        (a, e) => {
          a.update(e.j.toInt, a(e.j.toInt) + e.value)
          a
        },
        (a1, a2) => {
          for (i <- 0 until cooMatrix.numCols().toInt) {
            a1.update(i.toInt, a1(i.toInt) + a2(i.toInt))
          }
          a1
        })
  }

  /**
   * Sommes des éléments de chaque ligne d'une matrice.
   *
   * @param cooMatrix  la matrice sous forme de CoordinateMatrix
   * @param multiplyBy coefficient par lequel multiplier tous les éléments
   *                   (1 par défaut)
   * @return           un tableau des sommes des éléments de chaque ligne
   *                   multipliées par un coefficient
   */
  def rowSums(cooMatrix: CoordinateMatrix,
              multiplyBy: Double = 1): Array[Double] = {
    cooMatrix.entries.map(x => new MatrixEntry(x.i, x.j, x.value * multiplyBy))
      .aggregate(Array.fill[Double](cooMatrix.numRows().toInt)(0))(
        (a, e) => {
          a.update(e.i.toInt, a(e.i.toInt) + e.value)
          a
        },
        (a1, a2) => {
          for (i <- 0 until cooMatrix.numRows().toInt) {
            a1.update(i.toInt, a1(i.toInt) + a2(i.toInt))
          }
          a1
        })
  }

  /**
   * Créer une BlockMatrix ligne ou colonne à partir d'un Array.
   *
   * @param sc        un SparkContext
   * @param a         le tableau des valeurs
   * @param direction la direction de la matrice (ligne ou colonne) :
   *                  - v vertical
   *                  - h horizontal
   * @param blockSize la taille des blocs
   * @return          la matrice par blocs
   */
  def createBlockMatrixFromArray(
    sc: SparkContext, a: Array[Double], direction: Symbol,
    blockSize: Int): BlockMatrix = {
    val iterator = a.sliding(blockSize, blockSize)
    val iterator2 = if (direction == 'h) iterator.zipWithIndex
      .map(x => x match {
        case (array, index) => ((0, index),
          Matrices.dense(1, array.length, array))
      })
    else iterator.zipWithIndex.map(x => x match {
      case (array, index) => ((index, 0),
        Matrices.dense(array.length, 1, array))
    })
    val arr = iterator2.toArray
    val arr_p = sc.parallelize(arr)
    new BlockMatrix(arr_p, blockSize, blockSize)
  }

  /**
   * Créer une matrice W aléatoirement.
   *
   * @param sc        un SparkContext
   * @param nCols     nombre de colonnes (mots)
   * @param K         nombre de classes
   * @param blockSize taille des blocs de la matrice W
   * @return          la matrice W de taille nCols * K
   */
  def createRandomW(sc: SparkContext, nCols: Int, K: Int,
                    blockSize: Int): BlockMatrix = {
    val W_Entries = sc.parallelize((0 until nCols).map { i =>
      new MatrixEntry(i, scala.util.Random.nextInt(K), 1)
    })

    val W_coo = new CoordinateMatrix(W_Entries, nCols, K)

    W_coo.toBlockMatrix(blockSize, blockSize)
  }

  /**
   * Trouve le maximum et l'indice de la colonne correspondante de chaque ligne.
   *
   * @param m la matrice
   * @return  RDD de (i, (e, j)) avec :
   *          - i indice de la ligne
   *          - j indice de la colonne
   *          - e le maximum de la ligne i
   */
  def rows_max_and_argmax(
    m: BlockMatrix): RDD[(Long, (Double, Int))] = {
    m.toIndexedRowMatrix.rows.map(
      a => (a.index,
        a.vector.toArray.zipWithIndex.reduce((x, y) =>
          if (x._1 > y._1) x else y)))
  }

  /**
   * Créer la matrice d'assignation des colonnes ou des lignes aux classes.
   *
   * @param from      la matrice compressée à partir de laquelle déterminer les
   *                  classes
   * @param nRows     le nombre de lignes du résultat
   * @param nCols     le nombre de colonnes du résultat
   * @param blockSize la taille des blocs de la matrice à créer
   * @return          la matrice d'assignation aux classes
   */
  def create_Z_or_W(from: BlockMatrix, nRows: Int,
                    nCols: Int, blockSize: Int): BlockMatrix = {
    val entries = rows_max_and_argmax(from).map(x => x match {
      case (l, (e, c)) => new MatrixEntry(l, c, 1.0)
    })
    var Z_coo = new CoordinateMatrix(entries, nRows, nCols)
    Z_coo.toBlockMatrix(blockSize, blockSize)
  }

  /**
   * Calculer la trace d'une matrice par blocs.
   *
   * @param m la matrice
   * @return  la trace
   */
  def trace(m: BlockMatrix): Double = {
    // trace d'un bloc
    def tr(bm: Matrix): Double = {
      val n = bm.numCols.min(bm.numRows)
      (0 until n).map(i => bm(i, i)).reduce(_ + _)
    }
    // somme des traces des blocs
    m.blocks.map(b => b match {
      case ((i, j), bm) => if (i == j) tr(bm) else 0
    }).first()
  }

  /**
   * Créer la matrice B à partir de la matrice A.
   * 
   * @param cooMatrix la matrice A
   * @param blockSize la taille des blocs
   * @return          la matrice B
   */
  def createBFromA(cooMatrix: CoordinateMatrix, blockSize: Int): BlockMatrix = {
    val sc = cooMatrix.entries.sparkContext

    val A = cooMatrix.toBlockMatrix(blockSize, blockSize)

    val numCols = A.numCols().toInt
    val numRows = A.numRows().toInt

    val col_sums = ModularityWithBlockMatrix.colSums(cooMatrix)

    val N = col_sums.sum

    val minus_row_sums_over_N = ModularityWithBlockMatrix
      .rowSums(cooMatrix, -1 / N)

    val col_sums_v = ModularityWithBlockMatrix
      .createBlockMatrixFromArray(sc, col_sums, 'h, blockSize)

    val minus_row_sums_over_N_v = ModularityWithBlockMatrix
      .createBlockMatrixFromArray(sc, minus_row_sums_over_N, 'v, blockSize)

    val indep = minus_row_sums_over_N_v.multiply(col_sums_v)
    indep.add(A)
  }
}

