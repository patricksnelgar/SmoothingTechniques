package moa.smoothingtechniques;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BackgroundModel {

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
	   for (Map.Entry<String, Double> entry : this.probabilities.entrySet())
		   this.probabilities.put(entry.getKey(), entry.getValue() / this.totalCount);
   }
}
