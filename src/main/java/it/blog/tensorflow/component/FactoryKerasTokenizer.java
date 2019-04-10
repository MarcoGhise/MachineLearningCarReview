package it.blog.tensorflow.component;

import it.blog.tensorflow.textclassifier.SortedKerasTokenizer;

public class FactoryKerasTokenizer {

	private static SortedKerasTokenizer tokenizer = null;

	public static SortedKerasTokenizer getSortedKerasTokenizer(TrainFit fit) {
		if (tokenizer == null) {
			synchronized (FactoryKerasTokenizer.class) {
				if (tokenizer == null) {

					String[] itemsArray = fit.getRecords().toArray(new String[fit.getRecords().size()]);

					tokenizer = new SortedKerasTokenizer(fit.getNumberOfWords());

					tokenizer.fitOnTexts(itemsArray);
				}
			}
		}

		return tokenizer;
	}

}
