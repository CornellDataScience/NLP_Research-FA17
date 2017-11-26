import java.io.FileWriter

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql._
import org.apache.spark.sql.functions._


object Num_user_sentences {
  def main (args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Spark-1").master("local[*]").getOrCreate()
    import spark.implicits._

    val whywontthisbuild = true

    val review = spark.read.json("hdfs://localhost:9000/cai29/yelp/review.json")

      review.createTempView("rev")

      val split = udf((s: String) => s.split("(?<!\\w\\.\\w.)(?<![A-Z][a-z]\\.)(?<=\\.|\\?)\\s").length)
      spark.udf.register("split", split)

      val join = spark.sql(
        "SELECT user_id, split(text) AS text FROM rev")


      val a = join.groupBy(column("user_id")).agg(sum(column("text")))
      a.write.json("/home/cai29/user_text_sentences.json")

  }

}

