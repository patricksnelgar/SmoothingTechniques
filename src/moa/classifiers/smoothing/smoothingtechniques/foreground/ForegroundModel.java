package moa.classifiers.smoothing.smoothingtechniques.foreground;

import moa.classifiers.smoothing.smoothingtechniques.BackgroundModel;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;

import java.util.List;

public abstract class ForegroundModel {

	protected final BackgroundModel bm;
	protected final HistoryRetentionTechnique history;
	protected final double threshold;
	protected static final double log2 = Math.log(2);

	public ForegroundModel(final BackgroundModel bm,
						   final HistoryRetentionTechnique history,
						   final double threshold) {
		this.bm = bm;
		this.history = history;
		this.threshold = threshold;
	}

	/**
	 * Add the words in the supplied tweet to the current history technique,
	 * manages the max history requirements internally.
	 * @param tweet the tweet to add to the current history.
	 */
	public final void addTweet(List<String> tweet) {
		this.history.addTweet(tweet);
	}

	// The specific foreground models probability calculation.
	protected abstract double getProbability(String word);

	/**
	 * Calculates the perplexity for the the given tweet.
	 * @param tweet The tweet to calculate the perplexity for.
	 * @return The perplexity of the given tweet.
	 */
	public final double getPerplexity(List<String> tweet) {
		double sum = 0d;

		for (String w : tweet)
			sum += Math.log(this.getProbability(w));

		// Make it log base 2 and 1 / N
		sum /= (ForegroundModel.log2 * tweet.size());
		sum *= -1;

		return Math.pow(2, sum);
	}

	//todo:
	public final boolean getClassification(List<String> tweet) {
		double perplexity = this.getPerplexity(tweet);
		return perplexity > this.threshold;
	}

	/**
	 * Clears the current history.
	 */
	public final void reset() {
		if (this.bm != null)
			this.bm.reset();
		if (this.history != null)
			this.history.reset();
	}
}
