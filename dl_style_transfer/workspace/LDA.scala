import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer

import com.spotify.featran.numpy
import java.io.FileOutputStream
import java.io.FileWriter



object LDAforKenta {
  def main (args: Array[String]): Unit ={
    val spark = SparkSession.builder.appName("Spark-1").master("local[*]").getOrCreate()
    import spark.implicits._

    val review = spark.read.json("hdfs://localhost:9000/cai29/yelp/review.json")
    val business = spark.read.json("hdfs://localhost:9000/cai29/yelp/business.json")

    review.createTempView("rev")
    business.createTempView("bus")

    val split = udf((s : String) => s.split(" +"))
    spark.udf.register("split", split)

    val join = spark.sql(
      "SELECT categories, split(text) AS text FROM rev INNER JOIN bus ON rev.business_id = bus.business_id")

    join.filter(row => {
      row.getAs[Seq[String]]("categories").contains("Bars")
    }).show()

    val counts = new CountVectorizer()
        .setInputCol("text")
        .setVocabSize(10000)
        .setOutputCol("vecs")
        .fit(join.select($"text"))

    val transformed = counts.transform(join.select($"text"))

    val lda = new LDA()
        .setK(128)
        .setFeaturesCol("vecs")
        .fit(transformed)

    val vocab_out = new FileWriter("/home/cai29/vocab.txt")
    counts.vocabulary.foreach(s => vocab_out.write(s + "\n"))

    val out = new FileOutputStream("/home/cai29/lda_for_kenta.npy")
    numpy.NumPy.write[Double](out, new CustomIterator(lda.topicsMatrix.rowIter), 10000, 128)
  }

  class CustomIterator (it: Iterator[org.apache.spark.ml.linalg.Vector]) extends Iterator[Array[Double]] {
    override def hasNext: Boolean = it.hasNext

    override def next(): Array[Double] = it.next().toArray
  }

}
