package com.sgw.spark.analytics

import org.apache.spark.{SparkContext, SparkConf}

/**
 * author: steve
 */
object Covtype2 {
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

  import org.apache.spark.mllib.evaluation._
  import org.apache.spark.mllib.tree._
  import org.apache.spark.mllib.tree.model._
  import org.apache.spark.rdd._

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example => (model.predict(example.features), example.label))
    new MulticlassMetrics(predictionsAndLabels)
  }

  // find the best hyperparameters. have to have at least 40 bins since the soil feature ahs 40 distinct values
  // also specify the feature map indicating the number of distinct values for the wilderness and soil features
  val evaluations =
    for(impurity <- Array("gini", "entropy");
        depth    <- Array(10, 20, 30);
        bins     <- Array(40, 300))
      yield {
        val model = DecisionTree.trainClassifier(trainData, 7, Map(10 -> 4, 11 -> 40), impurity, depth, bins)
        val predictionAndLabels = cvData.map(example => (model.predict(example.features), example.label))
        val trainAccuracy = getMetrics(model, trainData).precision
        val cvAccurancy = getMetrics(model, cvData).precision
        ((impurity, depth, bins), (trainAccuracy, cvAccurancy))
      }

  evaluations.sortBy(_._2).reverse.foreach(println)

  // Results:
  //  ((gini,30,300),(0.9995850203082031,0.9356088114258049))
  //  ((gini,30,40),(0.9995140652313673,0.936127537434727))
  //  ((entropy,30,40),(0.9994044073853484,0.937614551993637))
  //  ((entropy,30,300),(0.9992130436932762,0.9424387038766123))
  //  ((gini,20,300),(0.970360129267249,0.9264100702009199))
  //  ((entropy,20,300),(0.9669650363483507,0.9252688729812912))
  //  ((gini,20,40),(0.966313539733768,0.925199709513435))
  //  ((entropy,20,40),(0.9648084320433127,0.9222602621295432))
  //  ((gini,10,300),(0.7967760593270449,0.7959158972230868))
  //  ((gini,10,40),(0.7893322267208219,0.7911436179410035))
  //  ((entropy,10,300),(0.7839073885736525,0.7842445620223398))
  //  ((entropy,10,40),(0.7781750784268614,0.7794031192724004))
  // So (gini,30,300) appears to be the best.
  // Now we need to test those parameters against the test data set.

  val model2 = DecisionTree.trainClassifier(trainData.union(cvData), 7, Map(10 -> 4, 11 -> 40), "gini", 30, 300)
  val predictionAndLabels2 = testData.map(example => (model2.predict(example.features), example.label))
  val accuracy2 = new MulticlassMetrics(predictionAndLabels2).precision

  // accuracy2: Double = 0.9402013942680093
  // so the hyperparameters appear to be good (not overfit) and the accuracy has improved from (91% in Covtype.scala)
  // to 94% (above)
}
