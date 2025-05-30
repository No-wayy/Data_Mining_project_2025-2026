import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

import java.util.Random;

public class RandomForestModel {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Data_Mining_project_2025-2026/Dataset/Healthcare-stroke-data_after_preprocess.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(4); // Choose class

        // Apply SMOTE with 300%
        SMOTE smote = new SMOTE();
        smote.setPercentage(300);
        smote.setInputFormat(dataset);
        Instances balancedDataset = Filter.useFilter(dataset, smote);

        // Build Random Forest
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100); // num of tree
        rf.buildClassifier(balancedDataset);

        // Save model
        SerializationHelper.write("Data_Mining_project_2025-2026/Model/RandomForest.model", rf);

        // Load model
        RandomForest loadedRF = (RandomForest) SerializationHelper.read("Data_Mining_project_2025-2026/Model/RandomForest.model");

        // Check model by Cross-validation
        long startTime = System.currentTimeMillis();
        Evaluation eval = new Evaluation(balancedDataset);
        eval.crossValidateModel(loadedRF, balancedDataset, 10, new Random(1));
        long endTime = System.currentTimeMillis();
        double runtimeSeconds = (endTime - startTime) / 1000.0;

        // Print result
        System.out.println("=== Results for Dataset with SMOTE (300%) - Random Forest ===");
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval.areaUnderROC(0));
        System.out.println("Kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
    }
}
