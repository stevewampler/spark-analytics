package com.sgw.spark.analytics

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation._

/**
 * author: steve
 */
object Music {
  lazy val conf = new SparkConf()
    .setAppName("Music")
    .setMaster("local[*]")
  lazy val sc = new SparkContext(conf)

  lazy val dir = "/Users/steve/profiledata_06-May-2005/"

  lazy val rawUserArtistData = sc.textFile(dir + "user_artist_data.txt")

  lazy val rawArtistData = sc.textFile(dir + "artist_data.txt")

  lazy val artistsById = rawArtistData.flatMap { line =>
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

  lazy val rawArtistAlias = sc.textFile(dir + "artist_alias.txt")

  lazy val artistAlias = rawArtistAlias.flatMap { line =>
    val tokens = line.split('\t')
    if (tokens(0).isEmpty) {
      None
    } else {
      Some((tokens(0).toInt, tokens(1).toInt))
    }
  }.collectAsMap()

  lazy val bArtistAlias = sc.broadcast(artistAlias)

  lazy val trainData = rawUserArtistData.map { line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    Rating(userID, finalArtistID, count)
  }.cache()

  lazy val model = {
    ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
  }

  def rawArtistsForUser(userId: Int) =
    rawUserArtistData.map(_.split(' ')).filter { case Array(user, _, _) => user.toInt == userId}

  def existingArtistIdsForUser(userId: Int) =
    rawArtistsForUser(userId).map { case Array(_, artist, _) => artist.toInt}.collect().toSet

  def existingArtistsForUser(userId: Int) = {
    val existingArtistIds = existingArtistIdsForUser(userId)
    artistsById.filter { case (id, name) => existingArtistIds.contains(id)}.values.collect()
  }

  def recommendedArtistIdsForUser(userId: Int, num: Int = 5) =
    model.recommendProducts(userId, num).map(_.product).toSet

  def recommendedArtistsForUser(userId: Int, num: Int = 5) = {
    val recommendedArtistIds = recommendedArtistIdsForUser(userId, num)
    artistsById.filter { case (id, name) => recommendedArtistIds.contains(id)}.values.collect()
  }

  def main(args: Array[String]) {
    val userId = 2093760

    println("Existing artists:")
    existingArtistsForUser(userId).foreach(println)

    println("Recommended artists:")
    recommendedArtistsForUser(userId).foreach(println)
  }
}
