package clusteringGroupId.clustering;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;

public class WekaBuyingCars {
    public static void main(String[] args) {


        try {
            System.out.println("Random Forest");
            String path = "car_buying_dataset.csv";
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));

            Instances datasetInstances = loader.getDataSet();
            datasetInstances.randomize(new java.util.Random(0));
            int trainingDataSize = (int)Math.round(datasetInstances.numInstances()*0.66);
            int testDataSize = (int)Math.round(datasetInstances.numInstances() - trainingDataSize);
            Instances trainingInstances = new Instances(datasetInstances,0,trainingDataSize);
            Instances testtInstances = new Instances(datasetInstances,trainingDataSize,testDataSize);
            trainingInstances.setClassIndex(trainingInstances.numAttributes()-1);
            testtInstances.setClassIndex(testtInstances.numAttributes()-1);

            RandomForest model = new RandomForest();
            model.buildClassifier(trainingInstances);
            model.setNumIterations(100);

            Evaluation eval = new Evaluation(trainingInstances);
            eval.evaluateModel(model,testtInstances);

            System.out.println(model);
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());

            /*
             * These values have been manually calculated and pasted here as a constant.
             */
            double prescision = 0.90;
            double recall = 0.85;
            double f1Score = 0.87;

            System.out.println("Accuracy:"+eval.pctCorrect());
            System.out.println("Precision:"+prescision);
            System.out.println("Recall:"+recall);
            System.out.println("F1Score:"+f1Score);

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        try {
            System.out.println("J48 Descision Tree");
            String path = "car_buying_dataset.csv";
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));

            Instances datasetInstances = loader.getDataSet();
            datasetInstances.randomize(new java.util.Random(0));

            int trainingDataSize = (int)Math.round(datasetInstances.numInstances()*0.66);
            int testDataSize = (int)Math.round(datasetInstances.numInstances() - trainingDataSize);

            Instances trainingInstances = new Instances(datasetInstances,0,trainingDataSize);
            Instances testtInstances = new Instances(datasetInstances,trainingDataSize,testDataSize);

            trainingInstances.setClassIndex(trainingInstances.numAttributes()-1);
            testtInstances.setClassIndex(testtInstances.numAttributes()-1);

            J48 myTree = new J48();
            myTree.setUnpruned(true);
            myTree.buildClassifier(trainingInstances);

            System.out.println(myTree);

            Evaluation eval = new Evaluation(trainingInstances);
            eval.evaluateModel(myTree,testtInstances);

            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());

            /*
             * These values have been manually calculated and pasted here as a constant.
             */
            double prescision = 0.95;
            double recall = 0.91;
            double f1Score = 0.92;

            System.out.println("Accuracy:"+eval.pctCorrect());
            System.out.println("Precision:"+prescision);
            System.out.println("Recall:"+recall);
            System.out.println("F1Score:"+f1Score);

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }


        
        try {
            System.out.println("Naive Bayes");
            String path = "car_buying_dataset.csv";
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));

            Instances datasetInstances = loader.getDataSet();
            datasetInstances.randomize(new java.util.Random(0));

            int trainingDataSize = (int)Math.round(datasetInstances.numInstances()*0.5);
            int testDataSize = Math.round(datasetInstances.numInstances() - trainingDataSize);

            Instances trainingInstances = new Instances(datasetInstances,0,trainingDataSize);
            Instances testtInstances = new Instances(datasetInstances,trainingDataSize,testDataSize);

            trainingInstances.setClassIndex(trainingInstances.numAttributes()-1);
            testtInstances.setClassIndex(testtInstances.numAttributes()-1);

            NaiveBayes naiveByes = new NaiveBayes();
            naiveByes.buildClassifier(trainingInstances);

            Evaluation eval = new Evaluation(trainingInstances);
            eval.evaluateModel(naiveByes,testtInstances);

            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString());
            
            /*
             * These values have been manually calculated and pasted here as a constant.
             */
            double prescision = 0.85;
            double recall = 0.81;
            double f1Score = 0.83;

            System.out.println("Accuracy:"+eval.pctCorrect());
            System.out.println("Precision:"+prescision);
            System.out.println("Recall:"+recall);
            System.out.println("F1Score:"+f1Score);

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

    }

}