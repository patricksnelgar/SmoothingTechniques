package moa.smoothingtechniques;

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
	 * @param tweet
	 */
	public final void addTweet(final List<String> tweet) {
		// If we have met our history cap, 'forget' the history.
		if (this.currentHistorySize == this.historySize)
			this.handleMaxHistorySize();
		this.currentHistorySize++;

		this.addInternalTweet(tweet);
	}

	protected void addInternalTweet(final List<String> tweet) {
		for (String word : tweet)
			this.addWord(word);
	}

	protected final void addWord(final String word) {
		Integer count = this.set.get(word);
		if (count == null) this.set.put(word, 1);
		else this.set.put(word, ++count);
		this.totalCount++;
	}

	protected abstract void handleMaxHistorySize();

	protected final Integer getWordCount(final String word) {
		return this.set.containsKey(word) ? this.set.get(word) : 0;
	}

	protected final int getTotalCount() {
		return this.totalCount;
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
