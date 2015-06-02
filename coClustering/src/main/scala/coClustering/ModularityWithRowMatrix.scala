package coClustering

import com.typesafe.config.ConfigFactory
import org.apache.spark.mllib.clustering.KMeans
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
  BlockMatrix,
  IndexedRowMatrix,
  DistributedMatrix
}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.DenseMatrix

class ModularityWithRowMatrix(
  var k: Int,
  maxIterations: Int = 30,
  blockSize: Int = 1024,
  epsilon: Double = 1e-9) {

  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  def run(cooMatrix: CoordinateMatrix): coClusteringModel = {

    // l'initialisation se fait par matrices par blocs

    val B_block_or_coo = ModularityWithRowMatrix.createBFromA(cooMatrix, blockSize)
    val a = B_block_or_coo match {
      case m: CoordinateMatrix => {
        m.entries.cache()
        val B = m.toIndexedRowMatrix()
        B.rows.cache()
        val Bt = m.transpose.toIndexedRowMatrix()
        Bt.rows.cache()
        m.entries.unpersist()
        (B, Bt)
      }
      case m: BlockMatrix => {
        m.blocks.cache()
        val B = m.toIndexedRowMatrix()
        B.rows.cache()
        val Bt = m.transpose.toIndexedRowMatrix()
        Bt.rows.cache()
        m.blocks.unpersist()
        (B, Bt)
      }
    }
    val B = a._1
    val Bt = a._2
    //val B_block = ModularityWithBlockMatrix.createBFromA(cooMatrix, blockSize)
    //    B_block.blocks.cache()

    //    val B = B_block_or_coo.toIndexedRowMatrix()
    //    B.rows.cache()

//    val Bt = B_block.transpose.toIndexedRowMatrix()
//    Bt.rows.cache()



    val sc = B.rows.sparkContext
    val numCols = B.numCols().toInt
    val numRows = B.numRows().toInt

    var W = ModularityWithRowMatrix.initW(sc, numCols, k, blockSize, Bt)

    var Z: Matrix = null
    var iter = 0

    var m_begin = Double.MinValue
    var change = true
    while (change & iter <= maxIterations) {
      change = false

      var BW = B.multiply(W)

      Z = ModularityWithRowMatrix
        .create_Z_or_W(BW, numRows, k, blockSize)

      var BtZ = Bt.multiply(Z)

      W = ModularityWithRowMatrix
        .create_Z_or_W(BtZ, numCols, k, blockSize)

      //TODO critère
      val BW_local = BW.toBlockMatrix.toLocalMatrix match {
        case m: DenseMatrix  => m
        case m: SparseMatrix => m.toDense
      }
      val k_times_k = Z.transpose.multiply(BW_local)
      var m_end = (0 until k).map(i => k_times_k(i, i)).reduce(_ + _)

      val diff = (m_end - m_begin)
      val diff_abs = if (diff < 0) -diff else diff
      if (diff_abs > epsilon) {
        println(diff_abs)
        println("epsilon")
        println(epsilon)
        m_begin = m_end
        change = true
        println("change")
      } else {
        println("change: false")
      }

      println(iter)
      println("Criterion: " + m_end)
      println("hi")
      println(iter)
      iter = iter + 1
    }

    val res = new coClusteringModel()
    res.setRowLabelsFromLocalMatrix(sc, Z)

    res

  }

}

object ModularityWithRowMatrix {
  def train(
    sc: SparkContext,
    cooMatrixFilePath: String,
    nRows: Int,
    nCols: Int,
    k: Int,
    maxIterations: Int = 30,
    blockSize: Int = 1024,
    epsilon: Double = 1e-9): coClusteringModel = {

    val A_coo = ModularityWithBlockMatrix.cooMatrix(sc, cooMatrixFilePath,
      nRows, nCols)

    new ModularityWithRowMatrix(k, maxIterations, blockSize, epsilon).run(A_coo)
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
    m: IndexedRowMatrix): RDD[(Long, (Double, Int))] = {
    m.rows.map(
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
  def create_Z_or_W(from: IndexedRowMatrix, nRows: Int,
                    nCols: Int, blockSize: Int): Matrix = {
    val entries = rows_max_and_argmax(from).map(x => x match {
      case (l, (e, c)) => new MatrixEntry(l, c, 1.0)
    })
    var Z_coo = new CoordinateMatrix(entries, nRows, nCols)
    Z_coo.toBlockMatrix(blockSize, blockSize).toLocalMatrix()
  }

  def initW(sc: SparkContext, numCols: Int, k: Int, blockSize: Int, Bt: IndexedRowMatrix): Matrix = {

    val config = ConfigFactory.load()
    val useKmeans = try {
      config.getBoolean("co-clustering.init.use-kmeans")
    } catch {
      case _: Throwable => false
    }

    if (useKmeans) {
      println("useKmeans")
      val kmeansMaxIterations = try {
        config.getInt("co-clustering.init.kmeans-max-iterations")
      } catch {
        case _: Throwable => 30
      }
      val kmeansRuns = try {
        config.getInt("co-clustering.init.kmeans-runs")
      } catch {
        case _: Throwable => 1
      }

      val rowVectors = Bt.rows.map {
        row => row.vector
      }.cache()
      val clusters = KMeans.train(rowVectors, k, kmeansMaxIterations, kmeansRuns)
      val labels = clusters.predict(rowVectors)
      rowVectors.unpersist(false)
      val entries = labels.zipWithIndex().map {
        e => MatrixEntry(e._2, e._1, 1.0)
      }
      new CoordinateMatrix(entries, numCols, k).toBlockMatrix().toLocalMatrix()
    } else {
      println("useKmeans: no")
      ModularityWithBlockMatrix
        .createRandomW(sc, numCols, k, blockSize).toLocalMatrix()
    }
  }

  def createBFromA(cooMatrix: CoordinateMatrix, blockSize: Int): DistributedMatrix = {
    val A = cooMatrix
    val config = ConfigFactory.load()
    val computeIndependence = try {
      config.getBoolean("co-clustering.init.compute-independence")
    } catch {
      case _: Throwable => false
    }
    if (!computeIndependence) {
      println("compute independence: skip")
      A
    } else {
      ModularityWithBlockMatrix.createBFromA(cooMatrix, blockSize)
    }
  }
}
