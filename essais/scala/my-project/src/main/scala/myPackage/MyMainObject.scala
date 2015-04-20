package myPackage

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.linalg.{Vector, Vectors, Matrices}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object MyMainObject extends App {
    val spConfig = (new SparkConf).setMaster("local").setAppName("SparkLA")
    val sc = new SparkContext(spConfig)

    val rows = sc.textFile("/data/matrice.txt").map { line =>
        val values = line.split(' ').map(_.toDouble)
        Vectors.sparse(values.length, values.zipWithIndex.map(e => (e._2, e._1)).filter(_._2 != 0.0))
    }
   
    val rmat = new RowMatrix(rows)
    
    // Build a local DenseMatrix
    val dm = sc.textFile("/data/matrice_locale.txt").map { line =>
        val values = line.split(' ').map(_.toDouble)
        Vectors.dense(values)
    }
    
    val ma = dm.map(_.toArray).take(dm.count.toInt)
    val localMat = Matrices.dense(dm.count.toInt,
                                  dm.take(1)(0).size,
                                  transpose(ma).flatten)

    // Multiply two matrices
    rmat.multiply(localMat).rows.foreach(println)

    /**
     * Transpose a matrix
     */
    def transpose(m: Array[Array[Double]]): Array[Array[Double]] = {
        (for {
            c <- m(0).indices
        } yield m.map(_(c)) ).toArray
    }
}
