package com.sgw.spark.analytics

import org.apache.spark.{SparkContext, SparkConf}

/**
 * author: steve
 */
object Music {
  val conf = new SparkConf()
    .setAppName("Music")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)

  import org.apache.spark.{SparkContext, SparkConf}
  import org.apache.spark.SparkContext._
  import org.apache.spark.mllib.recommendation._

  val dir = "s3://swampler/profiledata_06-May-2005/"

  val rawUserArtistData = sc.textFile(dir + "user_artist_data.txt")

  val rawArtistData = sc.textFile(dir + "artist_data.txt")

  val artistsById = rawArtistData.flatMap { line =>
    val (id, name) = line.span(_ != '\t')
    if (name.isEmpty) {
      None
    } else {
      try {
        Some((id.toInt, name.trim))
      } catch {
        case e: NumberFormatException => None
      }
    }
  }

  val rawArtistAlias = sc.textFile(dir + "artist_alias.txt")

  val artistAlias = rawArtistAlias.flatMap { line =>
    val tokens = line.split('\t')
    if (tokens(0).isEmpty) {
      None
    } else {
      Some((tokens(0).toInt, tokens(1).toInt))
    }
  }.collectAsMap()

  val bArtistAlias = sc.broadcast(artistAlias)

  val ratings = rawUserArtistData.map { line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    Rating(userID, finalArtistID, count)
  }.cache()

//  val model1 = ALS.trainImplicit(ratings, 10, 5, 0.01, 1.0)

  def rawArtistsForUser(userId: Int) =
    rawUserArtistData.map(_.split(' ')).filter { case Array(user, _, _) => user.toInt == userId}

  def existingArtistIdsForUser(userId: Int) =
    rawArtistsForUser(userId).map { case Array(_, artist, _) => artist.toInt}.collect().toSet

  def existingArtistsForUser(userId: Int) = {
    val existingArtistIds = existingArtistIdsForUser(userId)
    artistsById.filter { case (id, name) => existingArtistIds.contains(id)}.values.collect()
  }

  def recommendedArtistIdsForUser(model: MatrixFactorizationModel, userId: Int, num: Int = 5) =
    model.recommendProducts(userId, num).map(_.product).toSet

  def recommendedArtistsForUser(model: MatrixFactorizationModel, userId: Int, num: Int = 5) = {
    val recommendedArtistIds = recommendedArtistIdsForUser(model, userId, num)
    artistsById.filter { case (id, name) => recommendedArtistIds.contains(id)}.values.collect()
  }

//  val userId = 2093760

//  println("Existing artists:")
//  existingArtistsForUser(userId).foreach(println)
//
//  println("Recommended artists:")
//  recommendedArtistsForUser(model1, userId).foreach(println)

  // Code used to calculate the "area under the curve"

  import org.apache.spark.broadcast.Broadcast
  import org.apache.spark.mllib.recommendation._
  import org.apache.spark.rdd.RDD
  import scala.collection.mutable.ArrayBuffer
  import scala.util.Random

  def areaUnderCurve(
    positiveData: RDD[Rating],
    bAllItemIDs: Broadcast[Array[Int]],
    predictFunction: (RDD[(Int,Int)] => RDD[Rating])
  ) = {
    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive", and map to tuples
    val positiveUserProducts = positiveData.map(r => (r.user, r.product))
    // Make predictions for each of them, including a numeric score, and gather by user
    val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other items, excluding those that are "positive" for the user.
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
      // mapPartitions operates on many (user,positive-items) pairs at once
      userIDAndPosItemIDs => {
        // Init an RNG and the item IDs set once for partition
        val random = new Random()
        val allItemIDs = bAllItemIDs.value
        userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0
          // Keep about as many negative examples per user as positive.
          // Duplicates are OK
          while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.size))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }
          // Result is a collection of (user,negative-item) tuples
          negative.map(itemID => (userID, itemID))
        }
      }
    }.flatMap(t => t)
    // flatMap breaks the collections above down into one big set of tuples

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

    // Join positive and negative by user
    positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        // AUC may be viewed as the probability that a random positive item scores
        // higher than a random negative one. Here the proportion of all positive-negative
        // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
        var correct = 0L
        var total = 0L
        // For each pairing,
        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }
        // Return AUC: fraction of pairs ranked correctly
        correct.toDouble / total
    }.mean() // Return mean AUC over users
  }

  val Array(trainData, cvData) = ratings.randomSplit(Array(0.9, 0.1))
  trainData.cache()
  cvData.cache()

  val allItemIDs = ratings.map(_.product).distinct().collect()
  val bAllItemIDs = sc.broadcast(allItemIDs)

//  val model2 = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
//  val auc = areaUnderCurve(cvData, bAllItemIDs, model2.predict)

  def predictMostListened(sc: SparkContext, train: RDD[Rating])(allData: RDD[(Int,Int)]) = {
    val bListenCount =
      sc.broadcast(train.map(r => (r.product, r.rating)).reduceByKey(_ + _).collectAsMap())
    allData.map { case (user, product) =>
      Rating(user, product, bListenCount.value.getOrElse(product, 0.0))
    }
  }

  // val auc2 = areaUnderCurve(cvData, bAllItemIDs, predictMostListened(sc, trainData))

//  val evaluations = for(
//    rank <- Array(10, 50);
//    lambda <- Array(1.0, 0.0001);
//    alpha <- Array(1.0, 40.0)
//  ) yield {
//    val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
//    val auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
//    ((rank, lambda, alpha), auc)
//  }

//  evaluations.sortBy { case (_, auc) => auc }.reverse.foreach(println)

  // 50, 1, 40

  val model3 = ALS.trainImplicit(trainData, 50, 10, 1.0, 40.0)

  val someUsers = ratings.map(_.user).distinct().take(100)
  val someRecommendations = someUsers.map(userID => model3.recommendProducts(userID, 5))

  someRecommendations.map(
    recs => recs.head.user + " -> " + recs.map(_.product).mkString(", ")
  ).foreach(println)
}
