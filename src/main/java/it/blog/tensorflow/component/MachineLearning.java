package it.blog.tensorflow.component;

import java.io.IOException;

import javax.annotation.PostConstruct;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.TokenizerMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import it.blog.tensorflow.textclassifier.SortedKerasTokenizer;

@Component
public class MachineLearning {
	
	public static MultiLayerNetwork bowModel = null;
	
	@Autowired
	TrainFit trainFit;
	
	@PostConstruct
    private void init() throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
		String bowMlp = new ClassPathResource("model/bow.h5").getFile().getPath();
		bowModel = KerasModelImport.importKerasSequentialModelAndWeights(bowMlp);
	}
	
	public DataBuffer makePrediction(String sentence)
	{
		String[] texts = new String[] { sentence };
		
		SortedKerasTokenizer tokenizer = FactoryKerasTokenizer.getSortedKerasTokenizer(trainFit);
		INDArray input = tokenizer.textsToMatrix(texts, TokenizerMode.BINARY);

		INDArray result = bowModel.output(input);

		DataBuffer buffer = result.data();
		
		return buffer;
		
	}
}
