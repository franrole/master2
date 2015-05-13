package coClustering

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{
  BlockMatrix,
  IndexedRowMatrix
}
import org.apache.spark.mllib.linalg.{
  Vector,
  Vectors,
  Matrices,
  SparseMatrix,
  Matrix
}
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

class coClusteringModel {
  /**
   * les labels des lignes
   */
  var rowLabels: RDD[Int] = null

  /**
   * Définir les labels des lignes à partir d'une matrice par blocs (Z).
   *
   * @param Z une matrice par blocs
   */
  def setRowLabelsFromBlockMatrix(Z: BlockMatrix) {
    rowLabels = ModularityWithBlockMatrix.rows_max_and_argmax(Z)
      .sortBy(x => x._1).map(x => x._2._2).coalesce(1)
  }

  /**
   * Définir les labels des lignes à partir d'une matrice locale (Z).
   *
   * @param Z une matrice locale
   */
  def setRowLabelsFromLocalMatrix(sc: SparkContext, Z: Matrix) {
    val arr = Z.transpose.toArray
    val labels = for {
      i <- 0 until Z.numRows
      val l = arr.slice(i * Z.numCols, (i + 1) * Z.numCols - 1)
      val argmax = l.indexOf(l.max)
    } yield argmax
    rowLabels = sc.parallelize(labels, 1)
  }

  /**
   * Définir les labels des lignes à partir d'un graphe.
   * 
   * @param graph le graphe avec les nRows premiers sommets qui représentent les
   *              lignes
   * @param nRows le nombre de lignes (documents)
   */
  def setRowLabelsFromGraph(graph: Graph[Int, Double], nRows: Int) = {
    rowLabels = graph.vertices.filter {
      case (index, label) => index < nRows
    }.sortBy {
      case (index, label) => index
    }.map {
      case (index, label) => label
    }.coalesce(1)
  }
}