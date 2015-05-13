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
//  val path = "/home/stan/new/Stage/data/csv/"
    val path = "/data"

  def main(args: Array[String]) {
    val spConfig = (new SparkConf).setAppName("CoClustering")
    val sc = new SparkContext(spConfig)

    val corpus = "ng20"

    val (cooMatrixFilePath, nRows, nCols, k, predicted_labels_path) =
      getParamsForCorpus(corpus)

    //    val coClusteringModel = ModularityWithBlockMatrix
    //      .train(sc, cooMatrixFilePath, nRows, nCols, k)
    //    coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)

    val coClusteringModel = ModularityWithRowMatrix
      .train(sc, cooMatrixFilePath, nRows, nCols, k, 20, 1024, -1.0)
    coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)
    
//    val coClusteringModel = ModularityWithGraphX.train(sc, cooMatrixFilePath, nRows, nCols, k, 30, 1024, -1.0)
//    coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)
    

    //    val a = ModularityWithBlockMatrix.cooMatrix(sc, cooMatrixFilePath, nRows, nCols)
    //    val b = a.toRowMatrix()
    //    b.rows.count()
    //    println("rowMatrix")
    //
    //    val startTime = System.nanoTime()
    //    val pca = b.computePrincipalComponents(100)
    //    val timeInSeconds = (System.nanoTime() - startTime) / 1e9
    //
    //    val arr = pca.transpose.toArray
    //    val m = for {
    //      i <- 1 until pca.numRows
    //      val l = arr.slice((i - 1) * pca.numCols, i * pca.numCols - 1).mkString(",")
    //    } yield l
    //
    //    val r = m.mkString("\n")
    //
    //    println(r)
    //
    //    println("TIME")
    //    println(timeInSeconds)

  }

  /**
   * Obtenir les paramètres pour un corpus.
   *
   * @param corpus nom du corpus (cstr, classic3 ou ng20)
   * @return       (cooMatrixFilePath, nRows, nCols, K, rowsPredictedValuesPath)
   */
  def getParamsForCorpus(corpus: String): (String, Int, Int, Int, String) = {
    corpus match {
      case "cstr" => (s"${path}/cstr.csv",
        475, 1000, 4, s"${path}/predicted_labels/cstr")
      case "classic3" => (s"${path}/classic3.csv",
        3891, 4303, 3, s"${path}/predicted_labels/classic3")
      case "ng20" => (s"${path}/ng20.csv",
        19949, 43586, 20, s"${path}/predicted_labels_ng20_rowMatrix_20_iter")
    }
  }
}