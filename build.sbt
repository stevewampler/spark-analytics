name := "AdvancedAnalyticsWithSpark"

version := "1.0"

scalaVersion := "2.11.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.2.1",
  "org.apache.spark" %% "spark-mllib" % "1.2.1",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.4.1",
//  "edu.umd" % "cloud9" % "1.5.0",
//  "com.google.guava" % "guava" % "14.0.1",
//  "info.bliki.wiki" % "bliki-core" % "3.0.19",
//  "org.apache.hadoop" % "hadoop-client" %
  "com.cloudera.datascience" % "common" % "1.0.0"
)
