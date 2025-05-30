import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;

import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class NaiveBayesClassifier{
    public static void main(String[] args) throws Exception {
        DataSource source1 = new DataSource("Data_Mining_project_2025-2026/Dataset/Healthcare-stroke-data_after_preprocess.arff");
        Instances dataset1 = source1.getDataSet();

        dataset1.setClassIndex(4);

        // Apply SMOTE with 300%
        SMOTE smote = new SMOTE();
        smote.setPercentage(300); // 300% tăng cường dữ liệu thiểu số
        smote.setInputFormat(dataset1);
        Instances balancedDataset1 = Filter.useFilter(dataset1, smote);

        NaiveBayes bayes1 = new NaiveBayes();
        bayes1.buildClassifier(balancedDataset1);

        weka.core.SerializationHelper.write("Data_Mining_project_2025-2026/Model/Naive_Bayes.model", bayes1);

        NaiveBayes loadedBayes1 = (NaiveBayes) weka.core.SerializationHelper.read("Data_Mining_project_2025-2026/Model/Naive_Bayes.model");

        //Evaluation
        long startTime1 = System.currentTimeMillis(); // Record start time
        Evaluation eval1 = new Evaluation(balancedDataset1);
        eval1.crossValidateModel(loadedBayes1, balancedDataset1, 10, new Random(1));
        long endTime1 = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis1 = endTime1 - startTime1;
        double runtimeSeconds1 = runtimeMillis1 / 1000.0;

        System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval1.areaUnderROC(0));
        System.out.println("kappa = " + eval1.kappa());
        System.out.println("MAE = " + eval1.meanAbsoluteError());
        System.out.println("RMSE = " + eval1.rootMeanSquaredError());
        System.out.println("RAE = " + eval1.relativeAbsoluteError());
        System.out.println("RRSE = " + eval1.rootRelativeSquaredError());
        System.out.println("Error Rate = " + eval1.errorRate());
        System.out.println(eval1.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval1.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds1);

    }
}

