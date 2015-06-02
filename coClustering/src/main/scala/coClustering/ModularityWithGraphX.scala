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

import org.apache.spark.graphx._

class ModularityWithGraphX(
  var k: Int,
  maxIterations: Int = 30,
  blockSize: Int = 1024,
  epsilon: Double = 1e-9) {

  def run(g: Graph[Int, Double], nRows: Int, numCols: Int): coClusteringModel = {
    var graph = g
    val sc = graph.vertices.sparkContext

    //graph : arcs des mots vers les documents

    var wordsLabels = ModularityWithGraphX.createRandomLabels(graph, k, 'words)

    graph = ModularityWithGraphX.updateLabels(graph, wordsLabels)

    var docsLabels: VertexRDD[Int] = null
    var iter = 0
    var prevG: Graph[Int, Double] = null
    while (iter <= maxIterations) {
      //BW
      val docsFuzzyLabels = ModularityWithGraphX.fuzzyLabels(graph, k)
      docsFuzzyLabels.cache()

      //Z
      docsLabels = ModularityWithGraphX.getLabels(docsFuzzyLabels)
      docsLabels.cache()

      prevG = graph
      graph = ModularityWithGraphX.updateLabels(graph, docsLabels)

      //arcs des documents vers les mots
      graph = graph.reverse
      graph.cache()

      //BtZ
      val wordsFuzzyLabels = ModularityWithGraphX.fuzzyLabels(graph, k)
      wordsFuzzyLabels.cache()

      //W
      wordsLabels = ModularityWithGraphX.getLabels(wordsFuzzyLabels)
      wordsLabels.cache()

      graph = ModularityWithGraphX.updateLabels(graph, wordsLabels)

      //arcs des mots vers les documents
      graph = graph.reverse

      println(iter)
      ModularityWithGraphX.criterion(graph)
      prevG.unpersistVertices(blocking=true)
      prevG.edges.unpersist(blocking=true)
      docsFuzzyLabels.unpersist(blocking=true)
      wordsFuzzyLabels.unpersist(blocking=true)
      docsLabels.unpersist(blocking=true)
      wordsLabels.unpersist(blocking=true)      
      
      iter = iter + 1
    }

    var res = new coClusteringModel()
    res.setRowLabelsFromGraph(graph, nRows)
    res
  }
}

object ModularityWithGraphX {
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

    val B_coo = ModularityWithBlockMatrix.createBFromA(A_coo, blockSize).toCoordinateMatrix()

    val graph = createGraphFromCooMatrix(B_coo)

    new ModularityWithGraphX(k).run(graph, nRows, nCols)
  }

  /**
   * Créer un `Graph` à partir d'une `CoordinateMatrix`.
   *
   * @param m la matrice
   * @return  le graphe correspondant à la matrice, chaque arc partant d'une
   *          colonne (mot) et allant vers une ligne (document)
   */
  def createGraphFromCooMatrix(m: CoordinateMatrix): Graph[Int, Double] = {
    val nbRows = m.numRows()
    val edges: RDD[Edge[Double]] = m.entries.map {
      e => Edge(e.j + nbRows, e.i, e.value)
    }
    Graph.fromEdges(edges, defaultValue = 1)
  }

  /**
   * Créer des labels aléatoirement.
   *
   * @param graph le graphe
   * @param k     le nombre de classes
   * @param set   l'ensemble pour lequel créer les labels (si les arcs sont
   *              orientés des mots vers les documents) :
   *              - words pour les mots
   *              - docs  pour les documents
   * @return      les labels générés
   */
  def createRandomLabels(graph: Graph[Int, Double],
                         k: Int, set: Symbol): VertexRDD[Int] = {
    val vertices = if (set == 'docs) {
      graph.inDegrees.filter {
        case (id, inDegree) => inDegree > 0
      }
    } else {
      graph.outDegrees.filter {
        case (id, outDegree) => outDegree > 0
      }
    }
    vertices.mapValues[Int] { x: Int => scala.util.Random.nextInt(k) }
  }

  /**
   * Mettre à jour les labels dans le graphe.
   *
   * @param graph     le graphe
   * @param newLabels les labels à mettre à jour
   * @return          le graphe mis à jour avec les nouveaux labels
   */
  def updateLabels(graph: Graph[Int, Double],
                   newLabels: VertexRDD[Int]): Graph[Int, Double] = {
    graph.outerJoinVertices(newLabels) {
      (id, oldValue, newValue) => newValue.getOrElse(oldValue)
    }
  }

  /**
   * Classification floue non normalisée.
   *
   * @param graph le graphe
   * @param k     le nombre de classes
   * @return      somme des poids des arêtes arrivant à chaque vertex suivant la
   *              classe de la source
   */
  def fuzzyLabels(graph: Graph[Int, Double],
                  k: Int): VertexRDD[Array[Double]] = {
    //on n'utilise pas la valeur du nœud de destination
    val tripletFields = new TripletFields(true, false, true)
    graph.aggregateMessages[Array[Double]](
      triplet => { // Map Function
        //on envoie au nœud de destination un tableau représentant le nœud
        //source
        triplet.sendToDst(
          //            Array.tabulate(k) {
          //          //valeur de l'arc pour l'index correspondant au label du nœud source
          //          //0 sinon
          //          case (index) => if (index != triplet.srcAttr) 0 else triplet.attr
          //        }
          {
            val a = Array.fill(k) { 0.0 }
            a.update(triplet.srcAttr, triplet.attr)
            a
          })
      }, // Reduce Function
      //somme des deux tableaux suivant les classes
      (array1, array2) => {
        for (i <- 0 until k) {
          array2.update(i, array2(i) + array1(i))
        }
        array2
      },
      //array1.zip(array2).map { case (x, y) => (x + y) },
      tripletFields)
  }

  /**
   * Assignation des documents ou mots aux classes à partir d'une classification
   * floue.
   *
   * @param from la classification floue
   * @return     les labels des nœuds
   */
  def getLabels(from: VertexRDD[Array[Double]]) = {
    from.mapValues((id, value) => value match {
      case array => array.indexOf(array.max)
    })
  }

  /**
   * Calcul le critère d'arrêt.
   */
  def criterion(graph: Graph[Int, Double]): Double = {
    graph.triplets.map(e => if (e.srcAttr != e.dstAttr) e.attr else 0).sum
  }
}