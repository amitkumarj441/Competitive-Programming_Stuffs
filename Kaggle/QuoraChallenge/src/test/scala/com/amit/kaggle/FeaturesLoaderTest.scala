package com.amit.kaggle

import java.nio.file.{Files, Path}

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FlatSpec

class FeaturesLoaderTest extends FlatSpec with DataFrameSuiteBase {
  "FeaturesLoader" should "read train files" in {
    val trainFile = getClass.getResource("/quora/train100.csv").getFile
    val features = new FeaturesLoader().loadTrainFile(spark, trainFile).collect()
    assert(features.exists(row => Option(row.question2).isEmpty))  // nulls have not been cleaned
    assert(features.length === 100)
  }
  "FeaturesLoader" should "read test files" in {
    val testFile = getClass.getResource("/quora/test100.csv").getFile
    val features = new FeaturesLoader().loadTestFile(spark, testFile).collect()
    assert(features.map(_.id).distinct.length === features.length)
    assert(features.exists(r => Option(r.id).isEmpty))  // preserve brokenness of broken rows
  }
}
