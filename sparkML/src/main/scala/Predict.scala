import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.sql.{SQLContext , SparkSession}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Strategy;
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.collection.mutable;
import org.apache.spark.mllib.regression.LabeledPoint;

import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.storage.StorageLevel

object Predict {
   def main(args : Array[String]) {
      val conf = new SparkConf()
      val sc = new SparkContext(conf)
      val sqlContext = new org.apache.spark.sql.SQLContext(sc)
      val headerSchema = StructType(
        List(
      StructField("index", IntegerType, true),
      StructField("label", IntegerType, true)))
      
      val dfslist = mutable.ArrayBuffer[DataFrame]();
      val textFile1 = sc.textFile(args(0));
   
      // Load the test data 
      val testdata = textFile1.zipWithIndex().map({y => 
           val arr = y._1.split(",")
           val temp = new Array[String](arr.length-1); 
           Array.copy(arr, 0, temp, 0, arr.length-1)
           val tempk = temp.map(x => x.toDouble)
           // Here the label is just an index not really a label as I wanted to 
           // create a LabeledPoint type of data.
           new LabeledPoint(y._2.toInt,convertArrayToVector(tempk))
    })
      
    for(k <- 0 until 8){
      // Load the saved Model and running each model on test data.
      val pipelineread = RandomForestModel.load(sc,args(1) + "/sample-model" + k.toString());
      
      // predict labels on features
      val prediction = testdata.map({x =>
        val pred = pipelineread.predict(x.features)
        Row.fromTuple((x.label.toInt,pred.toInt))
      })
     
      // Load the creating label with the index into a dataframe and adding it to a list.
      dfslist += sqlContext.createDataFrame(prediction, headerSchema)
    }

      // Merging the list of data frames.
      val finalanswer = dfslist.tail.foldLeft(dfslist.head)((accDF, newDF) => accDF.join(newDF, Seq("index")))
      val finalanswersorted = finalanswer.sort("index")
      val temp = finalanswersorted.collect()
      
      // Only one reducer.
      val RDDdf = sc.parallelize(temp, 1)
      
      // Polling on all the Models.
      // Label is predicted by counting the labels for each trained data row.(i.e 0 or 1) whichever is more the label is predicted as that label.
      
      val result = RDDdf.map({x =>
        var count0 : Int  = 0 
        var count1 : Int  = 0 
       
        var predicted : Int  = 0
        for(j <- 1 until 9){
          val value = x.getInt(j)
          if(value.equals(0)){
            count0 += 1
          }
          else{
            count1 += 1
          }
        }
        if(count1 > count0){
          predicted = 1
        }
        predicted
        
 })
 
   // Saving the result
   result.saveAsTextFile(args(2))
      
   }
   def convertArrayToVector = ((features: Array[Double]) => Vectors.dense(features))
   
   
   
}