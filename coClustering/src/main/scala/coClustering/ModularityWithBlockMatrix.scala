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
    val sc = cooMatrix.entries.sparkContext
    
    val A = cooMatrix.toBlockMatrix(blockSize, blockSize)

    val col_sums = ModularityWithBlockMatrix.colSums(cooMatrix)

    val N = col_sums.sum

    val minus_row_sums_over_N = ModularityWithBlockMatrix
      .rowSums(cooMatrix, -1 / N)

    val col_sums_v = ModularityWithBlockMatrix
      .createBlockMatrixFromArray(sc, col_sums, "h", blockSize)

    val minus_row_sums_over_N_v = ModularityWithBlockMatrix
      .createBlockMatrixFromArray(sc, minus_row_sums_over_N, "v", blockSize)

    val indep = minus_row_sums_over_N_v.multiply(col_sums_v)

    val B = indep.add(A)

    var W = ModularityWithBlockMatrix
      .createRandomW(sc, A.numCols().toInt, k, blockSize)

    var m_begin = Double.MinValue
    var change = true

    var Z: BlockMatrix = null
    var iter = 0

    while (change & iter <= maxIterations) {
      change = false

      var BW = B.multiply(W)

      Z = ModularityWithBlockMatrix
        .create_Z_or_W(BW, A.numRows().toInt, k, blockSize)

      var BtZ = B.transpose.multiply(Z)

      W = ModularityWithBlockMatrix
        .create_Z_or_W(BtZ, A.numCols().toInt, k, blockSize)

      var k_times_k = Z.transpose.multiply(BW)

      var m_end = ModularityWithBlockMatrix.trace(k_times_k).first()

      val diff = (m_end - m_begin)
      val diff_abs = if (diff < 0) -diff else diff
      if (diff_abs > 1e-9) {
        m_begin = m_end
        change = true
      }
      println(iter)
      println(s"Criterion: $m_end")
      iter = iter + 1
    }

    val res = new coClusteringModel()
    res.setRowLabelsFromBlockMatrix(Z)

    res

  }

}

object ModularityWithBlockMatrix {
//  val path = "/home/stan/new/Stage/data/csv/"
  val path = "/data"

  /**
   * Obtenir les paramètres pour un corpus.
   *
   * @param corpus nom du corpus (cstr, classic3 ou ng20)
   * @return (cooMatrixFilePath, nRows, nCols, K, rowsPredictedValuesPath)
   */
  def getParamsForCorpus(corpus: String): (String, Int, Int, Int, String) = {
    corpus match {
      case "cstr" => (s"${path}/cstr.csv",
        475, 1000, 4, s"${path}/predicted_labels/cstr")
      case "classic3" => (s"${path}/classic3.csv",
        3891, 4303, 3, s"${path}/predicted_labels/classic3")
      case "ng20" => (s"${path}/ng20.csv",
        19949, 43586, 20, s"${path}/predicted_labels_ng20")
    }
  }

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

    new ModularityWithBlockMatrix(k).run(A_coo)
  }

  /**
   * Créer une CoordinateMatrix à partir d'un fichier
   */
  def cooMatrix(sc: SparkContext, filePath: String, nRows: Int,
                        nCols: Int): CoordinateMatrix = {
    val matrixEntries = sc.textFile(filePath).map { line =>
      val l = line.split(',')
      new MatrixEntry(l(0).toLong, l(1).toLong, l(2).toDouble)
    }

    new CoordinateMatrix(matrixEntries, nRows, nCols)
  }

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
   * Créer une BlockMatrix à partir d'un Array
   *
   * @param direction:
   *  - v vertical
   *  - h horizontal
   */
  def createBlockMatrixFromArray(
    sc: SparkContext, a: Array[Double], direction: String,
    blockSize: Int): BlockMatrix = {
    val iterator = a.sliding(blockSize, blockSize)
    val iterator2 = if (direction == "h") iterator.zipWithIndex
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
    val res = new BlockMatrix(arr_p, blockSize, blockSize)
    res
  }

  def createRandomW(sc: SparkContext, nCols: Int, K: Int,
                            blockSize: Int): BlockMatrix = {
    val W_Entries = sc.parallelize((0 until nCols).map { i =>
      new MatrixEntry(i, scala.util.Random.nextInt(K), 1)
    })

    val W_coo = new CoordinateMatrix(W_Entries, nCols, K)

    W_coo.toBlockMatrix(blockSize, blockSize)
  }

  def rows_max_and_argmax(
    m: BlockMatrix): RDD[(Long, (Double, Int))] = {
    m.toIndexedRowMatrix.rows.map(
      a => (a.index,
        a.vector.toArray.zipWithIndex.reduce((x, y) =>
          if (x._1 > y._1) x else y)))
  }

  def create_Z_or_W(from: BlockMatrix, nRows: Int,
                            nCols: Int, blockSize: Int): BlockMatrix = {
    val entries = rows_max_and_argmax(from).map(x => x match {
      case (l, (e, c)) => new MatrixEntry(l, c, 1.0)
    })
    var Z_coo = new CoordinateMatrix(entries, nRows, nCols)
    Z_coo.toBlockMatrix(blockSize, blockSize)
  }

  def trace(m: BlockMatrix): RDD[Double] = {
    def tr(bm: Matrix): Double = {
      val n = if (bm.numCols < bm.numRows) bm.numCols else bm.numRows
      val elements = for {
        i <- (0 until n)
        e = bm.apply(i, i)
      } yield e
      elements.reduce((a, b) => a + b)
    }
    m.blocks.map(b => b match {
      case ((i, j), bm) => if (i == j) tr(bm) else 0
    })
  }
}

