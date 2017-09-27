package com.amit.kaggle

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.amit.kaggle.FeaturesLoader.Features
import org.scalatest.FlatSpec

class CleanFeaturesTransformerTest extends FlatSpec with DataFrameSuiteBase {
  val features = Array(
    Features(id = 0, qid1 = 1, qid2 = null,
      question1 = "joao", question2 = "joão",
      isDuplicate = false),
    Features(id = null, qid1 = 1, qid2 = null,
      question1 = "joao", question2 = "joão",
      isDuplicate = false)
  )
  "CleanFeaturesTransformerTest" should "get rid of nulls and diacriticals" in {
    import spark.implicits._
    val transformer = new CleanFeaturesTransformer()
    val ds = spark.createDataset(features)
    val clean = transformer.transform(ds).as[Features].collect()
    assert(Option(clean.head.qid2).exists(_ === 0))  // filled null
    assert(clean.head.question1 === clean.head.question2)  // dropped diacriticals
    assert(clean.length === 1)
  }
}
