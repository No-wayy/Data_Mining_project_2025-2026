import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CSV2Arff{
    public static void main(String[]args) throws Exception{
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator("|||");
        loader.setSource(new File("Data_Mining_project_2025-2026/Dataset/Healthcare-stroke-data_after_preprocess.csv"));
        Instances data1 = loader.getDataSet();
        ArffSaver saver = new ArffSaver();


//        loader.setSource(new File("Loan_default.csv"));
//        Instances data1= loader.getDataSet();

        saver.setInstances(data1);
        saver.setFile(new File("Data_Mining_project_2025-2026/Dataset/Healthcare-stroke-data_after_preprocess.arff"));
        saver.writeBatch();
    }
}


