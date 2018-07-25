/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekajavatest;

import weka.core.Instances;
import weka.filters.Filter;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader ;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.FileInputStream;
import weka.filters.unsupervised.attribute.Remove;
 
import java.io.File;

/**
 *
 * @author Lenovo
 */
public class WekaJavaTest {

    /**
     * @param args the command line arguments
     */
    public static void ReadModel ()throws Exception
    {
        MultilayerPerceptron mp; 
        mp = (MultilayerPerceptron)(SerializationHelper.read(new FileInputStream("C:\\Users\\Lenovo\\Desktop\\Thesis\\Models\\MLP.model")));
        System.out.println(mp.toString());
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
      eval.crossValidateModel(j48, train, 10 , new Random(1));
      System.out.println(j48.prefix());
      System.out.println(eval.toSummaryString("\n Results \n=====\n",true));
        
    }
    public static void main(String[] args) throws Exception{
       // training
        //ReadModel();
       
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
      Remove removefilter = new Remove();
      int []remove = {0};
      removefilter.setAttributeIndicesArray(remove);
    //   removefilter.setInvertSelection(true);
        removefilter.setInputFormat(data);
        Instances newData = Filter.useFilter(train, removefilter);

      newData.setClassIndex(5);     
      reader.close();
       J48Test(newData);

      
  
     // System.out.println(eval.fMeasure(1)+" "+eval.precision(1)+" "+eval.recall(1)+" ");
      
    }
    
}
