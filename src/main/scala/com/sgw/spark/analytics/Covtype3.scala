package com.sgw.spark.analytics

import org.apache.spark.{SparkContext, SparkConf}

/**
 * author: steve
 */
object Covtype3 {
  val conf = new SparkConf()
    .setAppName("Music")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)

  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.regression.LabeledPoint

  val rawData = sc.textFile("s3://swampler/covtype.data")

  // convert the two categorical features from one-hot encoding to a series of distinct numeric values
  val data = rawData.map { line =>
    val values = line.split(',').map(_.toDouble)
    val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
    val soil = values.slice(14, 54).indexOf(1.0).toDouble
    val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
    val label = values.last - 1
    LabeledPoint(label, featureVector)
  }

  val Array(trainData, cvData, testData) =  data.randomSplit(Array(0.8, 0.1, 0.1))
  trainData.cache()
  cvData.cache()
  testData.cache()

  import org.apache.spark.mllib.tree._

  // create a RandomForest of DecisionTrees
  val forest = RandomForest.trainClassifier(trainData, 7, Map(10 -> 4, 11 -> 40), 20, "auto", "gini", 30, 300)

  // predict something
  val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
  val vector = Vectors.dense(input.split(',').map(_.toDouble))
  forest.predict(vector)
  // should be 4, which corresponds to class 5 "Aspen" (the original feature was 1-based)
}
