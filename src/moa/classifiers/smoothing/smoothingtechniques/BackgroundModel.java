package moa.classifiers.smoothing.smoothingtechniques;

import java.util.*;

public class BackgroundModel {

	private static final double VocabularyRestrictCount = 10d;
	private static final double sigma = 0.5d;

	// Stop words all lower case.
	private static final Set<String> StopWords = new HashSet<>(Arrays.asList(
			new String[] { "i", "a", "about", "an", "are", "as", "at", "be", "by", "com",
					"for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
					"when", "where", "who", "will", "with", "the", "www" }
	));

	private int totalCount;
	private final Map<String, Double> probabilities;

	public BackgroundModel() {
		this.probabilities = new HashMap<>();
		this.reset();
	}

	/**
	 * Build the Background models probabilities.
	 * @param words List of all words to load in from the background data set.
	 */
	public void buildProbabilities(List<String> words) {
		this.reset();
		for (String word : words) {
			//  Skip stop words in BM.
			if (!BackgroundModel.StopWords.contains((word)))
				this.addOccurrence(word);
		}
		this.calcProbabilities();
	}

	/**
	 * Gets the probability of the given word.
	 * @param word The word to get the probability of.
	 * @return The probability of the given word
	 */
	public Double getProbability(String word) {
		Double prob = this.probabilities.get(word);
		return prob == null ? 0f : prob;
	}

	/**
	 * Resets the background models probabilities.
	 */
	public void reset() {
		this.probabilities.clear();
		this.totalCount = 0;
	}

	protected void addOccurrence(String word) {
		Double count = this.probabilities.get(word);
		if (count == null) this.probabilities.put(word, 1d);
		else this.probabilities.put(word, count + 1);
		this.totalCount++;
	}

	protected void calcProbabilities() {
		double total = this.totalCount;
		for (Iterator<Map.Entry<String, Double>> it = this.probabilities.entrySet().iterator(); it.hasNext(); ) {
			Map.Entry<String, Double> entry = it.next();
			double wordCount = entry.getValue();
			if (wordCount <= BackgroundModel.VocabularyRestrictCount) {
				it.remove();
				continue;
			}
			double prob = Math.max(wordCount - BackgroundModel.sigma, 0d) / total;
			this.probabilities.put(entry.getKey(), prob);
		}
	}
}
