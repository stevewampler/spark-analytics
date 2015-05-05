package com.sgw.spark.analytics

import org.apache.spark.rdd.RDD
import org.apache.spark.util.StatCounter
import org.apache.spark.{SparkConf, SparkContext}
import java.lang.Double.isNaN

/**
 * author: steve
 */
object Linkage {
  def isHeader(line: String) = line.contains("id_1")

  def toDouble(s: String) = if ("?".equals(s)) Double.NaN else s.toDouble

  case class MatchData(id1: Int, id2: Int, scores: Array[Double], matched: Boolean)

  def parse(line: String) = {
    val pieces = line.split(',')
    val id1 = pieces(0).toInt
    val id2 = pieces(1).toInt
    val scores = pieces.slice(2, 11).map(toDouble)
    val matched = pieces(11).toBoolean
    MatchData(id1, id2, scores, matched)
  }


  class NAStatCounter extends Serializable {
    val stats: StatCounter = new StatCounter()
    var missing: Long = 0

    def add(x: Double): NAStatCounter = {
      if (java.lang.Double.isNaN(x)) {
        missing += 1
      } else {
        stats.merge(x)
      }
      this
    }

    def merge(other: NAStatCounter): NAStatCounter = {
      stats.merge(other.stats)
      missing += other.missing
      this
    }

    override def toString = {
      "stats: " + stats.toString + " NaN: " + missing
    }
  }

  object NAStatCounter extends Serializable {
    def apply(x: Double) = new NAStatCounter().add(x)
  }

  def statsWithMissing(rdd: RDD[Array[Double]]): Array[NAStatCounter] = {
    val nastats = rdd.mapPartitions((iter: Iterator[Array[Double]]) => {
      val nas: Array[NAStatCounter] = iter.next().map(d => NAStatCounter(d))
      iter.foreach(arr => {
        nas.zip(arr).foreach { case (n, d) => n.add(d) }
      })
      Iterator(nas)
    })
    nastats.reduce((n1, n2) => {
      n1.zip(n2).map { case (a, b) => a.merge(b) }
    })
  }

  def naz(d: Double) = if (java.lang.Double.isNaN(d)) 0.0 else d

  case class Scored(md: MatchData, score: Double)

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Linkage")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    val rawblocks = sc.textFile("/Users/steve/AdvanceAnalyticsWithSpark/rc/main/resources/linkage/blocks/")

    val noheader = rawblocks.filter(!isHeader(_))

    val parsed = noheader.map(line => parse(line))

    parsed.cache()

    val matchCounts = parsed.map(md => md.matched).countByValue()

    val matchCountsSeq = matchCounts.toSeq

    println("Match counts:")
    matchCountsSeq.sortBy(_._2).reverse.foreach(println)

    val nasRDD = parsed.map(_.scores.map(NAStatCounter(_)))

    val reduced = nasRDD.reduce((n1, n2) => n1.zip(n2).map { case (a, b) => a.merge(b) })

    println("Stats:")
    reduced.foreach(println)

    val statsm = statsWithMissing(parsed.filter(_.matched).map(_.scores))
    val statsn = statsWithMissing(parsed.filter(!_.matched).map(_.scores))

    val statsSummary = statsm.zip(statsn).map {
      case (m, n) => (m.missing + n.missing, m.stats.mean - n.stats.mean)
    }

    println("Stats Summary:")
    statsSummary.foreach(println)

    val ct = parsed.map(md => {
      val score = Array(2, 5, 6, 7, 8).map(i => naz(md.scores(i))).sum
      Scored(md, score)
    })

    val result1 = ct.filter(_.score >= 4.0).map(_.md.matched).countByValue()

    println("Result for cutoff = 4.0:")
    result1.foreach(println)

    val result2 = ct.filter(_.score >= 2.0).map(_.md.matched).countByValue()

    println("Result for cutoff = 2.0:")
    result2.foreach(println)
  }
}
