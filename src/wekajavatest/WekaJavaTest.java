/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekajavatest;

import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
import weka.core.DenseInstance;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Random;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.FileInputStream;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.functions.SMO;
 
import java.io.File;

/**
 *
 * @author Lenovo
 */
public class WekaJavaTest {

    /**
     * @param args the command line arguments
     */
   
    public static void ModelToText(String str) throws Exception 
    {
      BufferedWriter writer = new BufferedWriter(new FileWriter("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\model.csv"));
     String[] splited = str.split("\\s+");
     for(int i=0;i<splited.length;i++)
     {
         try{
         double value = Double.parseDouble(splited[i]);
         if(value==0||value==1||value==2||value==3||value==4||value==5||value==6||value==7)
         {
             continue;
             
         }
        else
         {
             writer.write(splited[i]);
             writer.newLine();
         }
         }
         catch(Exception e)
         {
             continue;
         }
     }
      writer.close();
      System.out.println("Success!");
}
    public static void ReadModel ()throws Exception
    {
        MultilayerPerceptron mp; 
        mp = (MultilayerPerceptron)(SerializationHelper.read(new FileInputStream("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\MLP.model")));
        System.out.print(mp.toString());
        ModelToText(mp.toString());
    }
    public static void ReadAnotherModel() throws Exception
    {
        SMO smo;
        smo = (SMO)(SerializationHelper.read(new FileInputStream("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\SVM.model")));
        System.out.println(smo.toString());
        ModelToText(smo.toString());
    }
    public static void ReadAthirdModel ()throws Exception
    {
        NaiveBayes nb; 
        nb = (NaiveBayes)(SerializationHelper.read(new FileInputStream("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\NB.model")));
        System.out.print(nb.toString());
        ModelToText(nb.toString());
    }
    public static void NNTest(Instances train) throws Exception
    {
        MultilayerPerceptron mp = new MultilayerPerceptron();
        mp.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
      eval.crossValidateModel(mp, train, 10 , new Random(1));
     System.out.println(mp.toString());
      System.out.println(eval.toSummaryString("\n Results \n=====\n",true));
    }
    public static void J48Test (Instances train)throws Exception
    {
        J48 j48 = new J48();
      j48.buildClassifier(train);
      Evaluation eval = new Evaluation(train);
      Instances test = TestSet();
      System.out.println(j48.prefix());
      //eval.evaluateModelOnce(j48, test.firstInstance());
      eval.evaluateModel(j48, test);
      System.out.println(eval.toSummaryString("\n Results \n=====\n",true));
      System.out.println("Predicted High and Actual High: "+eval.confusionMatrix()[0][0]);
      System.out.println("Predicted Low and Actual High: "+eval.confusionMatrix()[0][1]);
      System.out.println("Predicted High and Actual Low: "+eval.confusionMatrix()[1][0]);
      System.out.println("Predicted Low and Actual Low: "+eval.confusionMatrix()[1][1]);
      System.out.println(eval.toMatrixString());
      System.out.println(eval.toClassDetailsString());
    }
   public static Instances TestSet ()throws Exception
    {
        CSVLoader loader = new CSVLoader();
      loader.setSource(new File("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\J48Test.csv"));
      Instances data = loader.getDataSet();
       ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\J48Test.arff"));
    saver.writeBatch();
    data.setClassIndex(5);
    return data;
    }
    public static void main(String[] args) throws Exception{
       // training
      ReadAthirdModel();
      /*
      CSVLoader loader = new CSVLoader();
      loader.setSource(new File("C:\\Users\\Lenovo\\Desktop\\Thesis\\Data Sets\\Personality Train Dataset.csv"));
      Instances data = loader.getDataSet();
      data.setClassIndex(0);
       ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File("C:\\Users\\Lenovo\\Desktop\\Thesis\\Data Sets\\Personality Train Dataset.arff"));
    saver.writeBatch();
    
            BufferedReader reader = null;
      reader=new BufferedReader(new FileReader("C:\\Users\\Lenovo\\Desktop\\Thesis\\Data Sets\\Personality Train Dataset.arff"));
      Instances train =new Instances (reader);
      

      train.setClassIndex(5);     
      reader.close();
       J48Test(train);
*/
      
  
     // System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+" "+eval.recall(1)+" ");
      
    }
    
}
