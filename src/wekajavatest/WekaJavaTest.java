/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekajavatest;

import weka.core.Instances;
import weka.core.Instance;
import weka.filters.Filter;
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
   
    public static void ModelToText(String str,String container) throws Exception 
    {
      BufferedWriter writer = new BufferedWriter(new FileWriter(container));
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
    public static void ReadNNModel (String model, String container)throws Exception
    {
        MultilayerPerceptron mp; 
        mp = (MultilayerPerceptron)(SerializationHelper.read(new FileInputStream(model)));
        System.out.print(mp.toString());
        ModelToText(mp.toString(),container);
    }
    public static void ReadSVMModel(String model,String container) throws Exception
    {
        SMO smo;
        smo = (SMO)(SerializationHelper.read(new FileInputStream(model)));
        System.out.println(smo.toString());
        ModelToText(smo.toString(),container);
    }
  public static void ReadNBModel (String model, String container)throws Exception
    {
        NaiveBayes nb; 
        nb = (NaiveBayes)(SerializationHelper.read(new FileInputStream(model)));
        System.out.print(nb.toString());
        ModelToText(nb.toString(),container);
    }
   
    public static void main(String[] args) throws Exception{
       // training
       System.out.println("Reading Models from Neural Network, SVM, and Naive Bayes");
       System.out.println("Reading SVM Model"); 
      ReadSVMModel(args[0],args[1]);
      System.out.println("Reading Naive Bayes Model");
      ReadNBModel(args[2],args[3]);
      System.out.println("Reading Neural Network Model");
      ReadNNModel(args[4],args[5]);
      System.out.println("Finished Reading Models");
      /*
      
*/
      
  
     // System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+" "+eval.recall(1)+" ");
      
    }
    
}
