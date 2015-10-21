package moa.classifiers.smoothing.smoothingtechniques.foreground;

import moa.classifiers.smoothing.smoothingtechniques.BackgroundModel;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;

public class StupidBackoff extends ForegroundModel {

	protected final double alpha, normalizeCount, normalizeProbability;

	public StupidBackoff(final BackgroundModel bm,
						 final HistoryRetentionTechnique history,
						 final double threshold,
						 final double alpha) {
		super(bm, history, threshold);

		this.alpha = alpha;
		this.normalizeCount = 1f / (1f + alpha);
		this.normalizeProbability = alpha / (1f + alpha);
	}

	// Equivalent to the stupid backoff score but normalized into a probability.
	@Override
	protected final double getProbability(final String word) {
		int currWordCount = this.history.getWordCount(word);
		if (currWordCount == 0)
			return this.normalizeProbability * this.bm.getProbability(word);
		return this.normalizeCount * ((double)currWordCount / this.history.getAllWordsCounts());
	}

	/**
	 * Calculates the stupid backoff score for the given word.
	 * @param word The word to calculate the score for.
	 * @return The stupid backoff score for the given word.
	 */
	public final double getScore(final String word) {
		int currWordCount = this.history.getWordCount(word);
		if (currWordCount <= 0)
			return this.alpha * this.bm.getProbability(word);
		return ((double)currWordCount / this.history.getAllWordsCounts());
	}
}
