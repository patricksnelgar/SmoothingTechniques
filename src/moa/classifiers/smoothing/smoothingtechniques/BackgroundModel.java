package moa.classifiers.smoothing.smoothingtechniques;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class BackgroundModel {

	private static final int VocabularyRestrictCount = 0;//TODO: reset to higher number. 10;

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
		for (String word : words)
			this.addOccurrence(word);
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
			if (entry.getValue() <= BackgroundModel.VocabularyRestrictCount)
				it.remove();
			else
			// TODO: faster way than divide all values? have hash map of already computed divides? Dynamic programming?
				this.probabilities.put(entry.getKey(), (entry.getValue() / total));
		}
	}
}
