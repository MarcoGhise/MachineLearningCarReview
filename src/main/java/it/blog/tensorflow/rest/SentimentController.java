package it.blog.tensorflow.rest;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import it.blog.tensorflow.bean.Prediction;
import it.blog.tensorflow.bean.Summary;
import it.blog.tensorflow.component.MachineLearning;

@RestController
public class SentimentController {

	@Autowired
	MachineLearning ml;
	
	private static String[] factory = {"Dodge",	"Ferrari",	"Fiat",	"Ford",	"Genesis",	"Hummer",	"Hyundai",	"Infiniti",	"Isuzu",	"Jaguar",	"Jeep",	"Ki"};
	
	@RequestMapping(value="/review", method=RequestMethod.POST)
    public Summary greeting(@RequestBody String sentence) {
		
		Summary review = new Summary();
		
		DataBuffer buffer = ml.makePrediction(sentence);
		
		Prediction[] predictions = new Prediction[(int) buffer.length()];
		
		String percent = "0";
		for (int i=0; i< buffer.length(); i++) {
			percent =  String.valueOf(Math.round((buffer.getDouble(i)*100)*100)/100.0d);
			
			predictions[i] = new Prediction(factory[i], percent);
		}
		
		review.setPredictions(predictions);
		
		return review;
		
    }
}
