package myPackage

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.linalg.{ Vector, Vectors, Matrices, SparseMatrix, Matrix }
import org.apache.spark.mllib.linalg.distributed.{ MatrixEntry, CoordinateMatrix, BlockMatrix }
import org.apache.spark.rdd.RDD

object MyMainObject {
  def main(args: Array[String]) {
    val spConfig = (new SparkConf).setAppName("SparkLA")
    val sc = new SparkContext(spConfig)

    val blockSize = 1024

    /*
 * A
 */

    val corpus = "ng20"
    val max_iter = 30
    val path = "/data"

    def params(corpus: String): (String, Int, Int, Int, String) = {
      corpus match {
        case "cstr"     => (s"${path}/cstr.csv", 475, 1000, 4, s"${path}/predicted_labels/cstr")
        case "classic3" => (s"${path}/classic3.csv", 3891, 4303, 3, s"${path}/predicted_labels/classic3")
        case "ng20"     => (s"${path}/ng20.csv", 19949, 43586, 20, s"${path}/predicted_labels/ng20")
      }
    }

    val par = params(corpus)
    val A_Filename = par._1
    val A_nRows = par._2
    val A_nCols = par._3
    val K = par._4
    val predicted_labels_path = par._5

    val A_Entries = sc.textFile(A_Filename).map { line =>
      val l = line.split(',')
      new MatrixEntry(l(0).toLong, l(1).toLong, l(2).toDouble)
    }
    val A_coo = new CoordinateMatrix(A_Entries, A_nRows, A_nCols)
    val A = A_coo.toBlockMatrix(blockSize, blockSize)

    val col_sums = A_coo.entries.aggregate(Array.fill[Double](A_nCols)(0))(
      (a, e) => {
        a.update(e.j.toInt, a(e.j.toInt) + e.value)
        a
      },
      (a1, a2) => {
        for (i <- 0 until A_nCols) { a1.update(i.toInt, a1(i.toInt) + a2(i.toInt)) }
        a1
      })

    val N = col_sums.sum

    val minus_row_sums_over_N = A_coo.entries.aggregate(Array.fill[Double](A_nRows)(0))(
      (a, e) => {
        a.update(e.i.toInt, a(e.i.toInt) - e.value / N)
        a
      },
      (a1, a2) => {
        for (i <- 0 until A_nRows) { a1.update(i.toInt, a1(i.toInt) + a2(i.toInt)) }
        a1
      })

    //direction v ou h
    def createMatrixFromArray(a: Array[Double], direction: String): BlockMatrix = {
      val iterator = a.sliding(blockSize, blockSize)
      val iterator2 = if (direction == "h") iterator.zipWithIndex.map(x => x match { case (array, index) => ((0, index), Matrices.dense(1, array.length, array)) }) else iterator.zipWithIndex.map(x => x match { case (array, index) => ((index, 0), Matrices.dense(array.length, 1, array)) })
      val arr = iterator2.toArray
      val arr_p = sc.parallelize(arr)
      val res = new BlockMatrix(arr_p, blockSize, blockSize)
      res
    }

    //TODO découper Matrices.dense en plusieurs blocs
    //TODO on peut modifier les 2 dernières valeurs pour la taille des blocs
    //val col_sums_v = new BlockMatrix(sc.parallelize( Array(((0, 0), Matrices.dense(1, A_nCols, col_sums))) ), blockSize, blockSize)
    val col_sums_v = createMatrixFromArray(col_sums, "h")

    //val minus_row_sums_over_N_v = new BlockMatrix(sc.parallelize( Array(((0, 0), Matrices.dense(A_nRows, 1, minus_row_sums_over_N))) ), blockSize, blockSize)
    val minus_row_sums_over_N_v = createMatrixFromArray(minus_row_sums_over_N, "v")

    //TODO nb de blocs
    val indep = minus_row_sums_over_N_v.multiply(col_sums_v)

    val B = indep.add(A)

    /*
 * W
 */
    var W_Entries = sc.parallelize((0 until A_nCols).map { i =>
      new MatrixEntry(i, scala.util.Random.nextInt(K), 1)
    })

    var W_coo = new CoordinateMatrix(W_Entries, A_nCols, K)
    var W = W_coo.toBlockMatrix(blockSize, blockSize)

    var m_begin = Double.MinValue
    var change = true

    def rows_max_and_argmax(m: BlockMatrix): RDD[(Long, (Double, Int))] = {
      m.toIndexedRowMatrix.rows.map(a => (a.index, a.vector.toArray.zipWithIndex.reduce((x, y) => if (x._1 > y._1) x else y)))
    }

    def create_Z_or_W(from: BlockMatrix, nRows: Int, nCols: Int): BlockMatrix = {
      val entries = rows_max_and_argmax(from).map(x => x match { case (l, (e, c)) => new MatrixEntry(l, c, 1.0) })
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
      m.blocks.map(b => b match { case ((i, j), bm) => if (i == j) tr(bm) else 0 })
    }

    var Z: BlockMatrix = null
    var iter = 0
    while (change & iter <= max_iter) {
      change = false

      var BW = B.multiply(W)

      println("BW")

      Z = create_Z_or_W(BW, A_nRows, K)

      println("Z")

      var BtZ = B.transpose.multiply(Z)
      W = create_Z_or_W(BtZ, A_nCols, K)

      println("W")

      var k_times_k = Z.transpose.multiply(BW)

      println("k_times_k")

      var m_end = trace(k_times_k).first()

      println("trace")

      //abs method?
      val diff = (m_end - m_begin)
      val diff_abs = if (diff < 0) -diff else diff
      if (diff_abs > 1e-9) {
        m_begin = m_end
        change = true
      }
      println(s"Criterion: $m_end")
      iter = iter + 1
    }
    println(s"Iter: $iter")
    rows_max_and_argmax(Z).sortBy(x => x._1).map(x => x._2._2).coalesce(1).saveAsTextFile(predicted_labels_path)
    println(s"Iter: $iter")
  }
}