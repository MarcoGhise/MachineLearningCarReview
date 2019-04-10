package it.blog.tensorflow.component;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.annotation.PostConstruct;

import org.nd4j.linalg.io.ClassPathResource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;


@Component
public class TrainFit {

	public List<String> records = new ArrayList<>();
	
	@Value("${number_of_words}")	// 1000 words
	private Integer numberOfWords;
	
	@PostConstruct
    private void init() throws IOException {
		
		String fitontextPath = new ClassPathResource("model/fitontext.csv").getFile().getPath();

		try (BufferedReader br = new BufferedReader(new FileReader(fitontextPath))) {
			String line;
			String[] review;
			while ((line = br.readLine()) != null) {
				review = line.split("\\|");
				line = review[0].replaceAll("[;\\.\",]", " ");
				line = line.replaceAll("\\s{2}", " ");
				getRecords().add(line);
			}
		}
	}
	
	public Integer getNumberOfWords() {
		return numberOfWords;
	}

	public void setNumberOfWords(Integer numberOfWords) {
		this.numberOfWords = numberOfWords;
	}

	public List<String> getRecords() {
		return records;
	}

	public void setRecords(List<String> records) {
		this.records = records;
	}
}
