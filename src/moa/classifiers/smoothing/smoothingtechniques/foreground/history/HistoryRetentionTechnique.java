package moa.classifiers.smoothing.smoothingtechniques.foreground.history;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class HistoryRetentionTechnique {

	private final int historySize;

	protected int currentHistorySize, totalCount;
	protected final Map<String, Integer> set;

	public HistoryRetentionTechnique(final int historySize) {
		this.historySize = historySize;
		this.set = new HashMap<>(historySize);
		this.reset();
	}

	/**
	 * Add the words in the supplied tweet to the current history, manages the max
	 * history requirements internally.
	 * @param tweet the tweet to add to the current history.
	 */
	public final void addTweet(final List<String> tweet) {
		// If we have met our history cap, 'forget' the history.
		if (this.currentHistorySize == this.historySize)
			this.handleMaxHistorySize();
		this.currentHistorySize++;

		this.addInternalTweet(tweet);
	}

	protected abstract void addInternalTweet(final List<String> tweet);

	public final void addWord(final String word) {
		Integer count = this.set.get(word);
		if (count == null) this.set.put(word, 1);
		else this.set.put(word, count + 1);
		this.totalCount++;
	}

	protected abstract void handleMaxHistorySize();

	public final Integer getWordCount(final String word) {
		return this.set.containsKey(word) ? this.set.get(word) : 0;
	}

	public final int getAllWordsCounts() {
		return this.totalCount;
	}

	public final int getTotalUniqueWordCount() {
		return this.set.size();
	}

	/**
	 * Clears the current history.
	 */
	public void reset() {
		this.set.clear();
		this.totalCount = 0;
		this.currentHistorySize = 0;
	}
}
