package coClustering

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
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

object MainApplication {
  def main(args: Array[String]) {

    val spConfig = (new SparkConf).setAppName("CoClustering")
    val sc = new SparkContext(spConfig)

    val corpus = "ng20"

    val (cooMatrixFilePath, nRows, nCols, k, predicted_labels_path) =
      ModularityWithBlockMatrix.getParamsForCorpus(corpus)

    //    val coClusteringModel = ModularityWithBlockMatrix
    //      .train(sc, cooMatrixFilePath, nRows, nCols, k)

    //        coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)

    val a = ModularityWithBlockMatrix.cooMatrix(sc, cooMatrixFilePath, nRows, nCols)
    val b = a.toRowMatrix()
    b.rows.count()
    println("rowMatrix")

    val startTime = System.nanoTime()
    val pca = b.computePrincipalComponents(100)
    val timeInSeconds = (System.nanoTime() - startTime) / 1e9

    val arr = pca.transpose.toArray
    val m = for {
      i <- 1 until pca.numRows
      val l = arr.slice((i - 1) * pca.numCols, i * pca.numCols - 1).mkString(",")
    } yield l

    val r = m.mkString("\n")

    println(r)

    println("TIME")
    println(timeInSeconds)

  }
}