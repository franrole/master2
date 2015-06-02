package benchmark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{
  MatrixEntry,
  CoordinateMatrix
}
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.linalg.{ Matrices, Matrix }
import org.apache.spark.mllib.linalg.distributed.MatrixEntry

import org.apache.spark.rdd.RDD

object MainApplication {
  /** List of (title, time in ns) */
  val timeLogs = ListBuffer[(String, Long)]()

  def main(args: Array[String]) {

    val path = "/benchmark"
    //    val path = "/home/stan/new/Stage/benchmark/matrices"

    val spConfig = (new SparkConf).setAppName("Benchmark")
    val sc = new SparkContext(spConfig)

    val bigMatricesInfo = Array(
      (19949, 43586, "/data/ng20.csv"),
      (4000, 43586, "/data/ng20.csv"))

    val sparsity_list = Seq(80, 90, 99, 10)
    val nCols_list = Seq(3, 20, 100)

    //TODO
    //    val sparsity_list = Seq(90)
    //    val nCols_list = Seq(500, 1000)

    for (bigMatrixInfo <- bigMatricesInfo) {
      val bigMatrix = if (bigMatrixInfo._1 == 4000) {
        ng20_part(sc, bigMatrixInfo._3, bigMatrixInfo._1, bigMatrixInfo._2)
      } else {
        cooMatrix(sc, bigMatrixInfo._3, bigMatrixInfo._1, bigMatrixInfo._2)
      }
      bigMatrix.entries.cache()
      bigMatrix.entries.setName(s"ng20 ( ${bigMatrixInfo._1} first rows) entries")
      val local_nRows = bigMatrixInfo._2

      time("Count ng20 matrix entries") {
        bigMatrix.entries.count()
      }

      val bigRowMatrix = time("Create ng20 RowMatrix") {
        bigMatrix.toRowMatrix()
      }

      bigRowMatrix.rows.cache()
      bigRowMatrix.rows.setName("ng20 rows")
      time("Count ng20 rows") {
        bigRowMatrix.rows.count()
      }
      bigMatrix.entries.unpersist()

      for (sparsity <- sparsity_list) {
        for (local_nCols <- nCols_list) {
          //TODO
          val name = s"${local_nRows}_x_${local_nCols}_d_${sparsity}"
          //          val name = s"${local_nRows}_x_${local_nCols}_s_${sparsity}"

          val localMatrixPath = s"${path}/${name}.txt"
          val localCooMatrix = time(s"Count local cooMatrix entries ${name}") {
            val localCooMatrix = cooMatrix(sc, localMatrixPath, local_nRows, local_nCols)
            localCooMatrix.entries.cache()
            localCooMatrix.entries.setName(s"${name} entries")
            localCooMatrix.entries.count()
            localCooMatrix
          }
          //          val localMatrix = time(s"Create local cooMatrix ${name}") {
          //            localCooMatrix.toBlockMatrix().toLocalMatrix()
          //          }

          val localMatrix = time(s"Create local matrix ${name}") {
            mat(local_nRows, local_nCols, localCooMatrix.entries)
          }
          localCooMatrix.entries.unpersist()

          val m = time(s"Multiply ng20 by ${name}") {
            bigRowMatrix.multiply(localMatrix)
          }
          time(s"Count ng20 x ${name} rows") {
            m.rows.count()
          }

        }
      }
      bigRowMatrix.rows.unpersist()
    }

    println("*************************")
    timeLogs.toList.foreach { case (title, time) => println(s"${title}: ${time} ns (${time / 1e9} s)") }
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

  def ng20_part(sc: SparkContext, filePath: String, nRows: Int,
                nCols: Int): CoordinateMatrix = {
    val matrixEntries = sc.textFile(filePath).map { line =>
      val l = line.split(',')
      new MatrixEntry(l(0).toLong, l(1).toLong, l(2).toDouble)
    }.filter { x => x.i < nRows }

    new CoordinateMatrix(matrixEntries, nRows, nCols)
  }

  def mat(nRows: Int, nCols: Int, rdd: RDD[MatrixEntry]): Matrix = {
    val array = Array.fill(nRows * nCols) { 0.0 }
    rdd.collect().foreach { x => array(((x.i + 1) * (x.j + 1) - 1).toInt) = x.value }
    Matrices.dense(nRows, nCols, array)
  }

  /**
   * Mesurer le temps d'exécution d'un block
   */
  def time[R](title: String)(block: => R): R = {
    val t0 = System.nanoTime()
    val result = block // call-by-name
    val t1 = System.nanoTime()

    println(title)
    println("Elapsed time: " + (t1 - t0) + " ns (" + (t1 - t0) * 1e-9 + "s)")
    timeLogs.append((title, (t1 - t0)))

    result
  }

}