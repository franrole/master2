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
  var rowLabels: RDD[Int] = null

  def setRowLabelsFromBlockMatrix(Z: BlockMatrix) {
    def rows_max_and_argmax(m: BlockMatrix): RDD[(Long, (Double, Int))] = {
      m.toIndexedRowMatrix.rows.map(
        a => (a.index,
          a.vector.toArray.zipWithIndex.reduce((x, y) =>
            if (x._1 > y._1) x else y)))
    }

    rowLabels = rows_max_and_argmax(Z).sortBy(x => x._1).map(x => x._2._2)
      .coalesce(1)
  }

  def setRowLabelsFromLocalMatrix(sc: SparkContext, Z: Matrix) {
    val m = Z
    val arr = m.transpose.toArray
    val res = for {
      i <- 1 until m.numRows
      val l = arr.slice((i - 1) * m.numCols, i * m.numCols - 1)
      val argmax = l.indexOf(l.max)
    } yield argmax

    rowLabels = sc.parallelize(res, 1)
  }

  def setRowLabelsFromGraph(graph: Graph[Int, Double], nRows: Int) = {

    rowLabels = graph.vertices.filter {
      case (index, label) => index < nRows
    }.sortBy {
      case (index, label) => index
    }.map {
      case (index, label) => label
    }.coalesce(1)

    //    rowLabels = graph.outDegrees.filter {
    //      case (id, outDegree) => outDegree > 0
    //    }.map{
    //      case (index, outDegree) => index.toInt
    //    }.sortBy {
    //      case index => index
    //    }.coalesce(1)

  }
}