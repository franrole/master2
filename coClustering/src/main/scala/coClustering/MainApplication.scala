package coClustering

import com.typesafe.config.ConfigFactory
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
  val path = "/data"

  def main(args: Array[String]) {
    val spConfig = (new SparkConf).setAppName("CoClustering")
    val sc = new SparkContext(spConfig)
    val config = ConfigFactory.load()

    val corpus = "ng20"

    val cooMatrixFilePath = config.getString("co-clustering.data.input-matrix-path")
    val nRows = config.getInt("co-clustering.data.n-rows")
    val nCols = config.getInt("co-clustering.data.n-cols")
    val k = config.getInt("co-clustering.data.k")
    val maxIterations = config.getInt("co-clustering.algorithm.max-iterations")
    val epsilon = config.getDouble("co-clustering.algorithm.epsilon")
    //    val epsilon = 1e-9
    val predicted_labels_path = config.getString("co-clustering.results-path")
    val resultsPath = predicted_labels_path

    //    val (cooMatrixFilePath, nRows, nCols, k, predicted_labels_path) =
    //      getParamsForCorpus(corpus)

    val algorithm = try {
      config.getString("co-clustering.algorithm.name")
    } catch {
      case _: Throwable => "modularity"
    }

    algorithm match {
      case "compute-independence" => {
        println("algorithm: compute independence")
        val cooMatrix = ModularityWithBlockMatrix.cooMatrix(sc, cooMatrixFilePath, nRows, nCols)
        ModularityWithBlockMatrix.createBFromA(cooMatrix, 1024).toCoordinateMatrix().entries.map {
          e => s"${e.i},${e.j},${e.value}"
        }.coalesce(1).saveAsTextFile(resultsPath)
      }

      case "modularity" => {
        println("Modularity")
        val method = try {
          config.getString("co-clustering.algorithm.method")
        } catch {
          case _: Throwable => "graphx"
        }

        method match {
          case "rowMatrix" => {
            println("ROw Matrix")
            val coClusteringModel = ModularityWithRowMatrix
              .train(sc, cooMatrixFilePath, nRows, nCols, k, maxIterations, 1024, epsilon)
            coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)
          }

          case "blockMatrix" => {
            println("block Matrix")
            val coClusteringModel = ModularityWithBlockMatrix
              .train(sc, cooMatrixFilePath, nRows, nCols, k)
            coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)
          }

          case "graphx" => {
            println("graphxMatrix")
            val coClusteringModel = ModularityWithGraphX.train(sc, cooMatrixFilePath, nRows, nCols, k, 30, 1024, -1.0)
            coClusteringModel.rowLabels.saveAsTextFile(predicted_labels_path)
          }

          case _ => println("Unknown method: " + method)
        }
      }

      case _ => println("Unknown algorithm: " + algorithm)
    }

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