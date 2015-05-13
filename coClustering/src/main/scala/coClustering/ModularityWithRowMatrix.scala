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
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix

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
    val B_block = ModularityWithBlockMatrix.createBFromA(cooMatrix, blockSize)
    B_block.blocks.cache()

    val B = B_block.toIndexedRowMatrix()
    B.rows.cache()

    val Bt = B_block.transpose.toIndexedRowMatrix()
    Bt.rows.cache()

    B_block.blocks.unpersist()

    val sc = B.rows.sparkContext
    val numCols = B.numCols().toInt
    val numRows = B.numRows().toInt

    var W = ModularityWithBlockMatrix
      .createRandomW(sc, numCols, k, blockSize).toLocalMatrix()

    var Z: Matrix = null
    var iter = 0

    while (iter <= maxIterations) {

      var BW = B.multiply(W)

      Z = ModularityWithRowMatrix
        .create_Z_or_W(BW, numRows, k, blockSize)

      var BtZ = Bt.multiply(Z)

      W = ModularityWithRowMatrix
        .create_Z_or_W(BtZ, numCols, k, blockSize)

      //TODO critère

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

    new ModularityWithRowMatrix(k).run(A_coo)
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
}
