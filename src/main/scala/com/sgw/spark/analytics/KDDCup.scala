package com.sgw.spark.analytics

import org.apache.spark.{SparkContext, SparkConf}

/**
 * author: steve
 */
object KDDCup {
  val conf = new SparkConf()
    .setAppName("Music")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)

  // read the data
  val rawData = sc.textFile("s3://swampler/kddcup.data")

  // examine the data (count by network attack name)
  rawData.map(_.split(',').last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)

  // remove the three non-numeric columns and convert the rest to labeled data
  import org.apache.spark.mllib.linalg._
  import org.apache.spark.SparkContext._

  val labelsAndData = rawData.map { line =>
    val buffer = line.split(',').toBuffer
    buffer.remove(1, 3)
    val label = buffer.remove(buffer.length-1)
    val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
    (label, vector)
  }

  // k-means operates on just the feature vectors
  val data = labelsAndData.values.cache()

  import org.apache.spark.mllib.clustering._

  val kmeans = new KMeans()
  val model = kmeans.run(data) // defaults to 2 clusters

  // should print two vectors, really need more ...
  model.clusterCenters.foreach(println)

  // get a count the labels within each cluster
  val clusterLabelCount = labelsAndData.map { case (label,datum) =>
    val cluster = model.predict(datum)
    (cluster,label)
  }.countByValue

  clusterLabelCount.toSeq.sorted.foreach {
    case ((cluster, label), count) => println(f"$cluster%1s$label%18s$count%8s")
  }

  // Result:
  //  0             back.    2203
  //  0  buffer_overflow.      30
  //  0        ftp_write.       8
  //  0     guess_passwd.      53
  //  0             imap.      12
  //  0          ipsweep.   12481
  //  0             land.      21
  //  0       loadmodule.       9
  //  0         multihop.       7
  //  0          neptune. 1072017
  //  0             nmap.    2316
  //  0           normal.  972781
  //  0             perl.       3
  //  0              phf.       4
  //  0              pod.     264
  //  0        portsweep.   10412
  //  0          rootkit.      10
  //  0            satan.   15892
  //  0            smurf. 2807886
  //  0              spy.       2
  //  0         teardrop.     979
  //  0      warezclient.    1020
  //  0      warezmaster.      20
  //  1        portsweep.       1
  // output should show that the clustering wasn't good. only 1 data point ended up in cluster 1

  // define a euclidean distance function (sqrt of the sum of the squares of the difference between two vectors)
  def distance(a: Vector, b: Vector) = math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

  // and a function that returns the distance from a data point to its nearest cluster's centroid
  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }

  // define a function that measures the average distance to a centroid for a model build with a given k
  import org.apache.spark.rdd._

  def clusteringScore(data: RDD[Vector], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()
  }

  // now use the scoring function to evaluate values of k from 5 to 40
  (5 to 40 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)
}
