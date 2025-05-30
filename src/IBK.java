
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

import java.util.Random;

public class IBK {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source1 = new DataSource("Data_Mining_project_2025-2026/Dataset/Healthcare-stroke-data_after_preprocess.arff");
        Instances dataset1 = source1.getDataSet();
        dataset1.setClassIndex(4); // Choose class

        // Apply SMOTE with 300%
        SMOTE smote = new SMOTE();
        smote.setPercentage(300);
        smote.setInputFormat(dataset1);
        Instances balancedDataset1 = Filter.useFilter(dataset1, smote);

        // Build IBk
        IBk knn1 = new IBk();
        knn1.setKNN(5);
        knn1.buildClassifier(balancedDataset1);

        // Save model
        weka.core.SerializationHelper.write("Data_Mining_project_2025-2026/Model/IBK.model", knn1);

        // Load model
        IBk loadedKnn1 = (IBk) weka.core.SerializationHelper.read("Data_Mining_project_2025-2026/Model/IBK.model");

        // Check model by Cross-validation
        long startTime1 = System.currentTimeMillis();
        Evaluation eval1 = new Evaluation(balancedDataset1);
        eval1.crossValidateModel(loadedKnn1, balancedDataset1, 10, new Random(1));
        long endTime1 = System.currentTimeMillis();

        double runtimeSeconds1 = (endTime1 - startTime1) / 1000.0;

        // print result
        System.out.println("=== Results for Dataset 1 with SMOTE (300%) ===");
        System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval1.areaUnderROC(0));
        System.out.println("Kappa = " + eval1.kappa());
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
