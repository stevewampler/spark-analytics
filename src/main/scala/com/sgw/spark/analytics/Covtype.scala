package com.sgw.spark.analytics

import org.apache.spark.{SparkContext, SparkConf}

/**
 * author: steve
 */
object Covtype {
  val conf = new SparkConf()
    .setAppName("Music")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)

  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.regression.LabeledPoint

  val rawData = sc.textFile("s3://swampler/covtype.data")

  val data = rawData.map { line =>
    val values = line.split(',').map(_.toDouble)
    val featureVector = Vectors.dense(values.init)
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

  val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), "gini", 4, 100)

  val metrics = getMetrics(model, cvData)

  // look at the confusion matrix
  metrics.confusionMatrix

  // summarize the accuracy (should be about 70%, is that good or not?)
  metrics.precision

  // compute precision and recall for each category
  (0 until 7).map(cat => (metrics.precision(cat), metrics.recall(cat))).foreach(println)

  // what about a random classifier based on the class prevalence in the training set (should be about 37%, so 70%
  // accuracy seems pretty good after all)
  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
    val countsByCategory = data.map(_.label).countByValue()
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }

  val trainPriorProbabilities = classProbabilities(trainData)
  val cvPriorProbabilities = classProbabilities(cvData)
  trainPriorProbabilities.zip(cvPriorProbabilities).map {
    case (trainProb, cvProb) => trainProb * cvProb
  }.sum

  // now we need to find better hyperparameters without overfitting
  val evaluations =
    for(impurity <- Array("gini", "entropy");
        depth    <- Array(1, 20);
        bins     <- Array(10, 300))
      yield {
        val model = DecisionTree.trainClassifier(trainData, 7, Map[Int,Int](), impurity, depth, bins)
        val predictionAndLabels = cvData.map(example => (model.predict(example.features), example.label))
        val accuracy = new MulticlassMetrics(predictionAndLabels).precision
        ((impurity, depth, bins), accuracy)
      }

  evaluations.sortBy(_._2).reverse.foreach(println)

  // Results:
  //  ((entropy,20,300),0.9090550978813777)
  //  ((gini,20,300),0.9090550978813777)
  //  ((entropy,20,10),0.8961926456231696)
  //  ((gini,20,10),0.8895644578416428)
  //  ((gini,1,10),0.631578947368421)
  //  ((gini,1,300),0.6303457961532533)
  //  ((entropy,1,300),0.4885847877095929)
  //  ((entropy,1,10),0.4885847877095929)
  // So (entropy, 20, 300) appear to be the best hyperparameters.
  // Now we need to test those parameters against the test data set.

  val model2 = DecisionTree.trainClassifier(trainData.union(cvData), 7, Map[Int,Int](), "entropy", 20, 300)
  val predictionAndLabels2 = testData.map(example => (model2.predict(example.features), example.label))
  val accuracy2 = new MulticlassMetrics(predictionAndLabels2).precision

  // accuracy2: Double = 0.9070351326718371
  // so the hyperparameters appear to be good (not overfit)

  // what's the accuracy of the model when run over the data used to train it?
  val predictionsAndLabels3 = trainData.union(cvData).map(example => (model2.predict(example.features), example.label))
  val accuracy3 = new MulticlassMetrics(predictionsAndLabels3).precision
  // accuracy3: Double = 0.9459779941514093
  // the slightly higher accuracy may indicat theat the desicion tree is overfit the training data to some extent.
  // a lower max. depth on the decision tree might be a better choice.
}
