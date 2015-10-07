package moa.smoothingtechniques;

import java.util.List;

public abstract class ForegroundModel {

	protected final BackgroundModel bm;
	protected final HistoryRetentionTechnique history;

	public ForegroundModel(final BackgroundModel bm, final HistoryRetentionTechnique history) {
		this.bm = bm;
		this.history = history;
	}

	/**
	 * Add the words in the supplied tweet to the current history technique,
	 * manages the max history requirements internally.
	 * @param tweet
	 */
	public final void addTweet(List<String> tweet) {
		this.history.addTweet(tweet);
	}

	// The specific foreground models' probability calculation.
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

		sum /= tweet.size();
		sum *= -1;

		return Math.pow(2, sum);
	}

	/**
	 * Clears the current history.
	 */
	public final void reset() {
		this.bm.reset();
		this.history.reset();
	}
}
