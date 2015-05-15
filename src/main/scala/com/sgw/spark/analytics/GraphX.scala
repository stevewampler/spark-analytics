package com.sgw.spark.analytics


import org.apache.spark.{SparkContext, SparkConf}

/**
 * author: steve
 */
object GraphX {
  val conf = new SparkConf()
    .setAppName("Music")
    .setMaster("local[*]")
  val sc = new SparkContext(conf)

  import scala.xml._
  import com.cloudera.datascience.common.XmlInputFormat
  import org.apache.spark.SparkContext
  import org.apache.hadoop.io.{Text, LongWritable}
  import org.apache.hadoop.conf.Configuration
  import org.apache.spark.rdd.RDD

  def loadMedline(sc: SparkContext, path: String): RDD[String] = {
    val conf = new Configuration()
    conf.set(XmlInputFormat.START_TAG_KEY, "<MedlineCitation ")
    conf.set(XmlInputFormat.END_TAG_KEY, "</MedlineCitation>")
    val in = sc.newAPIHadoopFile(path, classOf[XmlInputFormat],
      classOf[LongWritable], classOf[Text], conf)
    in.map(line => line._2.toString)
  }

  def majorTopics(elem: Elem): Seq[String] = {
    val dn = elem \\ "DescriptorName"
    val mt = dn.filter(n => (n \ "@MajorTopicYN").text == "Y")
    mt.map(n => n.text)
  }

  val medlineRaw = loadMedline( sc, "s3://swampler/medline")
  val mxml: RDD[Elem] = medlineRaw.map(XML.loadString)
  val medline: RDD[Seq[String]] = mxml.map(majorTopics).cache()
}
