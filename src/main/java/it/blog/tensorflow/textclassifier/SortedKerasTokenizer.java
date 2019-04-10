package it.blog.tensorflow.textclassifier;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.TokenizerMode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils.parseJsonString;

public class SortedKerasTokenizer {

	private static final String DEFAULT_FILTER = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n";
    private static final String DEFAULT_SPLIT = " ";

    private Integer numWords;
    private String filters;
    private boolean lower;
    private String split;
    private boolean charLevel;
    private String outOfVocabularyToken;

    private LinkedHashMap<String, Integer> wordCounts = new LinkedHashMap<>();
    private LinkedHashMap<String, Integer> wordDocs = new LinkedHashMap<>();
    private Map<String, Integer> wordIndex = new HashMap<>();
    private Map<Integer, String> indexWord = new HashMap<>();
    private Map<Integer, Integer> indexDocs = new HashMap<>();
    private Integer documentCount;



    /**
     * Create a Keras Tokenizer instance with full set of properties.
     *
     * @param numWords             The maximum vocabulary size, can be null
     * @param filters              Characters to filter
     * @param lower                whether to lowercase input or not
     * @param split                by which string to split words (usually single space)
     * @param charLevel            whether to operate on character- or word-level
     * @param outOfVocabularyToken replace items outside the vocabulary by this token
     */
    public SortedKerasTokenizer(Integer numWords, String filters, boolean lower, String split, boolean charLevel,
                     String outOfVocabularyToken) {

        this.numWords = numWords;
        this.filters = filters;
        this.lower = lower;
        this.split = split;
        this.charLevel = charLevel;
        this.outOfVocabularyToken = outOfVocabularyToken;
    }


    /**
     * Tokenizer constructor with only numWords specified
     *
     * @param numWords             The maximum vocabulary size, can be null
     */
    public SortedKerasTokenizer(Integer numWords) {
        this(numWords, DEFAULT_FILTER, true, DEFAULT_SPLIT, false, null);
    }

    /**
     * Default Keras tokenizer constructor
     */
    public SortedKerasTokenizer() {
        this(null, DEFAULT_FILTER, true, DEFAULT_SPLIT, false, null);
    }


    /**
     * Import Keras Tokenizer from JSON file created with `tokenizer.to_json()` in Python.
     *
     * @param jsonFileName Full path of the JSON file to load
     * @return Keras Tokenizer instance loaded from JSON
     * @throws IOException I/O exception
     * @throws InvalidKerasConfigurationException Invalid Keras configuration
     */
    public static SortedKerasTokenizer fromJson(String jsonFileName) throws IOException, InvalidKerasConfigurationException {
        String json = new String(Files.readAllBytes(Paths.get(jsonFileName)));
        Map<String, Object> tokenizerBaseConfig = parseJsonString(json);
        Map<String, Object> tokenizerConfig;
        if (tokenizerBaseConfig.containsKey("config"))
            tokenizerConfig = (Map<String, Object>) tokenizerBaseConfig.get("config");
        else
            throw new InvalidKerasConfigurationException("No configuration found for Keras tokenizer");


        Integer numWords = (Integer) tokenizerConfig.get("num_words");
        String filters = (String) tokenizerConfig.get("filters");
        Boolean lower = (Boolean) tokenizerConfig.get("lower");
        String split = (String) tokenizerConfig.get("split");
        Boolean charLevel = (Boolean) tokenizerConfig.get("char_level");
        String oovToken = (String) tokenizerConfig.get("oov_token");
        Integer documentCount = (Integer) tokenizerConfig.get("document_count");

        @SuppressWarnings("unchecked")
        LinkedHashMap<String, Integer> wordCounts = (LinkedHashMap) parseJsonString((String) tokenizerConfig.get("word_counts"));
        @SuppressWarnings("unchecked")
        Map<String, Integer> wordDocs = (Map) parseJsonString((String) tokenizerConfig.get("word_docs"));
        @SuppressWarnings("unchecked")
        Map<String, Integer> wordIndex = (Map) parseJsonString((String) tokenizerConfig.get("word_index"));
        @SuppressWarnings("unchecked")
        Map<Integer, String> indexWord = (Map) parseJsonString((String) tokenizerConfig.get("index_word"));
        @SuppressWarnings("unchecked")
        Map<Integer, Integer> indexDocs = (Map) parseJsonString((String) tokenizerConfig.get("index_docs"));

        SortedKerasTokenizer tokenizer = new SortedKerasTokenizer(numWords, filters, lower, split, charLevel, oovToken);
        tokenizer.setDocumentCount(documentCount);
        tokenizer.setWordCounts(wordCounts);
        tokenizer.setWordDocs(new LinkedHashMap<>(wordDocs));
        tokenizer.setWordIndex(wordIndex);
        tokenizer.setIndexWord(indexWord);
        tokenizer.setIndexDocs(indexDocs);

        return tokenizer;
    }

    /**
     * Turns a String text into a sequence of tokens.
     *
     * @param text                 input text
     * @param filters              characters to filter
     * @param lower                whether to lowercase input or not
     * @param split                by which string to split words (usually single space)
     * @return Sequence of tokens as String array
     */
    public static String[] textToWordSequence(String text, String filters, boolean lower, String split) {
        if (lower)
            text = text.toLowerCase();

        for (String filter: filters.split("")) {
            text = text.replace(filter, split);
        }
        String[] sequences = text.split(split);
        
        List<String> seqList = new ArrayList<>(Arrays.asList(sequences));
        List<String> removeItem = Arrays.asList("");

        seqList.removeAll(removeItem);        

        return seqList.toArray(new String[seqList.size()]);
    }

    /**
     * Fit this tokenizer on a corpus of texts.
     *
     * @param texts array of strings to fit tokenizer on.
     */
    public void fitOnTexts(String[] texts) {
        String[] sequence;
        for (String text : texts) {
            if (getDocumentCount() == null)
                setDocumentCount(1);
            else
                setDocumentCount(getDocumentCount() + 1);
            if (charLevel) {
                if (lower)
                    text = text.toLowerCase();
                sequence = text.split("");
            } else {
                sequence = textToWordSequence(text, filters, lower, split);
            }
            for (String word : sequence) {
                if (getWordCounts().containsKey(word))
                    getWordCounts().put(word, getWordCounts().get(word) + 1);
                else
                    getWordCounts().put(word, 1);
            }
            Set<String> sequenceSet = new LinkedHashSet<String>(Arrays.asList(sequence));
            for (String word: sequenceSet) {
                if (getWordDocs().containsKey(word))
                    getWordDocs().put(word, getWordDocs().get(word) + 1);
                else
                    getWordDocs().put(word, 1);
            }
        }

        LinkedHashMap<String, Integer> sorted = wordCounts
                .entrySet()
                .stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(
                		Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2,
                        LinkedHashMap::new));
        

        int index = 1;
        for(String key : sorted.keySet()){
        	getWordIndex().put(key, index);
        	index++;
        }        
        
        for(String key : getWordIndex().keySet()){
            getIndexWord().put(getWordIndex().get(key), key);
        }

        for (String key: getWordDocs().keySet())
            getIndexDocs().put(getWordIndex().get(key), getWordDocs().get(key));
    }

    /**
     * Fit this tokenizer on a corpus of word indices
     *
     * @param sequences array of indices derived from a text.
     */
    public void fitOnSequences(Integer[][] sequences) {
        setDocumentCount(getDocumentCount() + 1);
        for (Integer[] sequence: sequences) {
            Set<Integer> sequenceSet = new HashSet<>(Arrays.asList(sequence));
            for (Integer index: sequenceSet)
                getIndexDocs().put(index, getIndexDocs().get(index) + 1);
        }
    }

    /**
     * Transforms a bunch of texts into their index representations.
     *
     * @param texts input texts
     * @return array of indices of the texts
     */
    public Integer[][] textsToSequences(String[] texts) {
        Integer oovTokenIndex  = getWordIndex().get(outOfVocabularyToken);
        String[] wordSequence;
        ArrayList<Integer[]> sequences = new ArrayList<>();
        for (String text: texts) {
            if (charLevel) {
                if (lower) {
                    text = text.toLowerCase();
                }
                wordSequence = text.split("");
            } else {
                wordSequence = textToWordSequence(text, filters, lower, split);
            }
            ArrayList<Integer> indexVector = new ArrayList<>();
            for (String word: wordSequence) {
                if (getWordIndex().containsKey(word)) {
                    int index = getWordIndex().get(word);
                    if (numWords != null && index >= numWords) {
                        if (oovTokenIndex != null)
                            indexVector.add(oovTokenIndex);
                    } else {
                        indexVector.add(index);
                    }
                } else if (oovTokenIndex != null) {
                    indexVector.add(oovTokenIndex);
                }
            }
            Integer[] indices = indexVector.toArray(new Integer[indexVector.size()]);
            sequences.add(indices);
        }
        return sequences.toArray(new Integer[sequences.size()][]);
    }


    /**
     * Turns index sequences back into texts
     *
     * @param sequences index sequences
     * @return text reconstructed from sequences
     */
    public String[] sequencesToTexts(Integer[][] sequences) {
        Integer oovTokenIndex  = getWordIndex().get(outOfVocabularyToken);
        ArrayList<String> texts = new ArrayList<>();
        for (Integer[] sequence: sequences) {
            ArrayList<String> wordVector = new ArrayList<>();
            for (Integer index: sequence) {
                if (getIndexWord().containsKey(index)) {
                    String word = getIndexWord().get(index);
                    if (numWords != null && index >= numWords) {
                        if (oovTokenIndex != null) {
                            wordVector.add(getIndexWord().get(oovTokenIndex));
                        } else {
                            wordVector.add(word);
                        }
                    }
                } else if (oovTokenIndex != null) {
                    wordVector.add(getIndexWord().get(oovTokenIndex));
                }
            }
            StringBuilder builder = new StringBuilder();
            for (String word: wordVector) {
                builder.append(word + split);
            }
            String text = builder.toString();
            texts.add(text);
        }
        return texts.toArray(new String[texts.size()]);
    }


    /**
     * Turns an array of texts into an ND4J matrix of shape
     * (number of texts, number of words in vocabulary)
     *
     * @param texts input texts
     * @param mode TokenizerMode that controls how to vectorize data
     * @return resulting matrix representation
     */
    public INDArray textsToMatrix(String[] texts, TokenizerMode mode) {
        Integer[][] sequences = textsToSequences(texts);
        return sequencesToMatrix(sequences, mode);
    }

    /**
     * Turns an array of index sequences into an ND4J matrix of shape
     * (number of texts, number of words in vocabulary)
     *
     * @param sequences input sequences
     * @param mode TokenizerMode that controls how to vectorize data
     * @return resulting matrix representatio
     */
    public INDArray sequencesToMatrix(Integer[][] sequences, TokenizerMode mode) {
        if (numWords == null) {
            if (!getWordIndex().isEmpty()) {
                numWords = getWordIndex().size();
            } else {
                throw new IllegalArgumentException("Either specify numWords argument" +
                        "or fit Tokenizer on data first, i.e. by using fitOnTexts");
            }
        }
        if (mode.equals(TokenizerMode.TFIDF) && getDocumentCount() == null) {
            throw new IllegalArgumentException("To use TFIDF mode you need to" +
                    "fit the Tokenizer instance with fitOnTexts first.");
        }
        INDArray x = Nd4j.zeros(sequences.length, numWords);
        for (int i=0; i< sequences.length; i++) {
            Integer[] sequence = sequences[i];
            if (sequence == null)
                continue;
            HashMap<Integer, Integer> counts = new HashMap<>();
            for (int j: sequence) {
                if (j >= numWords)
                    continue;
                if (counts.containsKey(j))
                    counts.put(j, counts.get(j) + 1);
                else
                    counts.put(j, 1);
            }
            for (int j: counts.keySet()) {
                int count = counts.get(j);
                switch (mode) {
                    case COUNT:
                        x.put(i, j, count);
                        break;
                    case FREQ:
                        x.put(i, j, count / sequence.length);
                        break;
                    case BINARY:
                        x.put(i, j, 1);
                        break;
                    case TFIDF:
                        double tf = 1.0 + Math.log(count);
                        int index = getIndexDocs().containsKey(j) ? getIndexDocs().get(j) : 0;
                        double idf = Math.log(1 + getDocumentCount() / (1.0 + index));
                        x.put(i, j, tf * idf);
                        break;
                }
            }
        }
        return x;
    }


	public Map<String, Integer> getWordCounts() {
		return wordCounts;
	}


	public void setWordCounts(LinkedHashMap<String, Integer> wordCounts) {
		this.wordCounts = wordCounts;
	}


	public Integer getDocumentCount() {
		return documentCount;
	}


	public void setDocumentCount(Integer documentCount) {
		this.documentCount = documentCount;
	}


	public LinkedHashMap<String, Integer> getWordDocs() {
		return wordDocs;
	}


	public void setWordDocs(LinkedHashMap<String, Integer> wordDocs) {
		this.wordDocs = wordDocs;
	}


	public Map<Integer, String> getIndexWord() {
		return indexWord;
	}


	public void setIndexWord(Map<Integer, String> indexWord) {
		this.indexWord = indexWord;
	}


	public Map<String, Integer> getWordIndex() {
		return wordIndex;
	}


	public void setWordIndex(Map<String, Integer> wordIndex) {
		this.wordIndex = wordIndex;
	}


	public Map<Integer, Integer> getIndexDocs() {
		return indexDocs;
	}


	public void setIndexDocs(Map<Integer, Integer> indexDocs) {
		this.indexDocs = indexDocs;
	}
}
