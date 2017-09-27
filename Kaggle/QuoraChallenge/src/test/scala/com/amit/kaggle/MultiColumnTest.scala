package com.amit.kaggle

import java.io.File
import java.nio.file.{Files, Path, Paths}

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.feature.Tokenizer
import org.scalatest.FlatSpec

class MultiColumnPipelineTest extends FlatSpec with DataFrameSuiteBase {
  "MultiColumnPipelineTest" should "serialize back and forth" in {
    import spark.implicits._
    val tmpdir: Path = Files.createTempDirectory("mcpipelinetest")
    val tokenizerDir: Path = Paths.get(tmpdir.toString, "ds")
    val tokenizer  = new MultiColumnPipeline()
        .setInputCols(Array("question1", "question2"))
        .setOutputCols(Array("question1tok", "question2tok"))
        .setStage(new Tokenizer)
    tokenizer.save(tokenizerDir.toString)
    val loaded = MultiColumnPipeline.read.load(tokenizerDir.toString)
    assert(loaded.getInputCols.head === "question1")
    val df = sc.makeRDD(Seq(("the q1", "q2 is last"))).toDF("question1", "question2")
    val tok = loaded.fit(df).transform(df)
    val q1tokens = tok.select("question1tok").as[Array[String]].collect.head
    assert(q1tokens === Array("the", "q1"))
    new File(tmpdir.toUri).delete()
  }
}
