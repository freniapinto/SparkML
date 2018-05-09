
import Array._
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

import org.apache.spark.mllib.tree.DecisionTree

import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.storage.StorageLevel

object Model {
   def main(args : Array[String]) {
    val conf = new SparkConf();
    
    val sc = new SparkContext(conf);
    val textFile = sc.textFile(args(0));    
  
    // Read from the training data 
    val data = textFile.map(x => {
      val arr = x.split(",").map(x => x.toDouble)
      val temp = new Array[Double](arr.length-1);
      Array.copy(arr, 0, temp, 0, arr.length-1)
      val label = arr(arr.length-1).toInt;
      (convertArrayToVector(temp),label)})
    
    for(l <- 0 until 8){ 
     val strategy = Strategy.defaultStrategy("Classification")
     val numTrees = 20
     val featureSubsetStrategy = "auto"
     val categoricalFeaturesInfo = Map[Int,Int]()
     val maxdepth : Int = 30
     val mxbins : Int = 32
     val seeds: Int = 12345
      if(l == 0){
        
        // Transform the training set for more training data
        val transform = data.mapPartitions({x =>
          val result : Array[Double] = ofDim[Double](3087)
          x.map(({y => 
            val a = rotate90(y,result)
            new LabeledPoint(a._1,a._2)  
          
          }))}, true)
          
          // Create a model using the above data
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          // Save the model for predicting
          model.save(sc,args(1) + "/sample-model" + l.toString() )
      }
     
      else if(l == 1) {
        
        val transform = data.mapPartitions({x =>
          val result : Array[Double] = ofDim[Double](3087)
        
          x.map(({y => 
            val a = rotate270(y,result)
            new LabeledPoint(a._1,a._2)  
          
          }))}, true)
          
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          model.save(sc,args(1) + "/sample-model" + l.toString() )
      }
     
      else if(l == 2) {
        val transform = data.mapPartitions({x =>
          val result : Array[Double] = ofDim[Double](3087)
          x.map(({y => 
            val a = rotate180(y,result)
            new LabeledPoint(a._2,a._1)  
          
          }))}, true)
          
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          model.save(sc,args(1) + "/sample-model" + l.toString() )
      }
     
      else if(l == 3) {
        val transform = data.mapPartitions({x =>
          val result : Array[Double] = ofDim[Double](3087)
          x.map(({y => 
            val a = mirror(y,result)
            new LabeledPoint(a._1,a._2)  
          
          }))}, true)
          
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          model.save(sc,args(1) + "/sample-model" + l.toString() )
      }
      else if(l == 4) {
        val transform = data.mapPartitions({x =>
          val result : Array[Double] = ofDim[Double](3087)
          x.map(({y =>
            val rotate90temp = rotate90(y,result)
            val a = mirror((rotate90temp._2,rotate90temp._1),result)
            new LabeledPoint(a._1,a._2)  
          
          }))}, true)
          
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          model.save(sc,args(1) + "/sample-model" + l.toString())
      }
      else if(l == 5) {
         val transform = data.mapPartitions({x =>
          val result : Array[Double] = ofDim[Double](3087)
          x.map(({y =>
            val rotate90temp = rotate270(y,result)
            val a = mirror((rotate90temp._2,rotate90temp._1),result)
            new LabeledPoint(a._1,a._2)  
          
          }))}, true)
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          model.save(sc,args(1) + "/sample-model" + l.toString() )
      }
      else if(l == 6) {
        val result : Array[Double] = ofDim[Double](3087)
        val transform = data.mapPartitions(x =>
          x.map(({y =>
           
            val a = mirror(rotate180(y,result),result)
            new LabeledPoint(a._1,a._2)  
          
          })), true)
          
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
          model.save(sc,args(1) + "/sample-model" + l.toString() )
      }
      else{
        val transform = data.mapPartitions(x => x.map(y => new LabeledPoint(y._2,y._1)))
          val model  =  RandomForest.trainClassifier(transform,strategy,numTrees,featureSubsetStrategy,seeds)
        model.save(sc,args(1) + "/sample-model" + l.toString() )
    }
      
    }
   

   }
 
   // rotate data by 90
  def rotate90(row : (org.apache.spark.mllib.linalg.Vector, Int),result:Array[Double]) : (Int,org.apache.spark.mllib.linalg.Vector) = {
    val arrtemp : Array[Double] = row._1.toArray
    for(k <- 0 until arrtemp.length){
        val kfactor = k / 441
        val i = (k % 441) / 21
        val j = (k % 441) % 21
        val newj90 : Int = ((i - 10) * -1) + 10
        val newk90 = (j * 21) + newj90 + (kfactor * 441)
        result(newk90) = arrtemp(k)
    }
    
    (row._2,convertArrayToVector(result))
  }
  
  def rotate180(row : (org.apache.spark.mllib.linalg.Vector, Int),result:Array[Double]) : (org.apache.spark.mllib.linalg.Vector, Int) = {
    
    val arrtemp : Array[Double] = row._1.toArray
    val result : Array[Double] = ofDim[Double](arrtemp.length) 
    for(k <- 0 until arrtemp.length){
        val kfactor = k / 441
        val i = (k % 441) / 21
        val j = (k % 441) % 21
        val newj180 : Int = ((j - 10) * -1) + 10
        val newi180 : Int = ((i - 10) * -1) + 10
        val newk180 = (newi180 * 21) + newj180 + (kfactor * 441)
        result(newk180) = arrtemp(k)
  }
    
    (convertArrayToVector(result),row._2)
  }
  
  // rotate data by 270
  def rotate270(row : (org.apache.spark.mllib.linalg.Vector, Int),result:Array[Double]) :  (Int,org.apache.spark.mllib.linalg.Vector) = {
    
    val arrtemp : Array[Double] = row._1.toArray
    val result : Array[Double] = ofDim[Double](arrtemp.length)
    for(k <- 0 until arrtemp.length){
        val kfactor = k / 441
        val i = (k % 441) / 21
        val j = (k % 441) % 21
        val newi270 : Int = ((j - 10) * -1) + 10
        val newk270 = (newi270 * 21) + i + (kfactor * 441)
        result(newk270) = arrtemp(k)
  }
    
    (row._2,convertArrayToVector(result))
  }
  
   // mirroring data
  def mirror(row : (org.apache.spark.mllib.linalg.Vector, Int),result:Array[Double]) : (Int,org.apache.spark.mllib.linalg.Vector) = {
    
    val arrtemp : Array[Double] = row._1.toArray
    val result : Array[Double] = ofDim[Double](arrtemp.length)
    for(k <- 0 until arrtemp.length) {
        val kfactor = k / 441
        val i = (k % 441) / 21
        val j = (k % 441) % 21
        val newimir : Int = ((i - 10) * -1) + 10
        val newkmir = (newimir * 21) + j + (kfactor * 441)
        result(newkmir) = arrtemp(k)
  }
    (row._2,convertArrayToVector(result))
  }
  
  // Converting Array to Vector
  def convertArrayToVector = ((features: Array[Double]) => Vectors.dense(features))
    
}