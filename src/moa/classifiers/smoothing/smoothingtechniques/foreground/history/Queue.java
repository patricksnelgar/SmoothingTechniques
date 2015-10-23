package moa.classifiers.smoothing.smoothingtechniques.foreground.history;

import java.util.LinkedList;
import java.util.List;

public class Queue extends HistoryRetentionTechnique {

	protected final java.util.Queue<List<String>> queue;

	public Queue(final int historySize) {
		super(historySize);
		this.queue = new LinkedList<>();
	}

	@Override
	protected void addInternalTweet(List<String> tweet) {
		this.queue.add(tweet);
		for (String word : tweet)
			this.addWord(word);
	}

	/**
	 * Remove the oldest history item when the max history is reached.
	 */
	@Override
	protected void handleMaxHistorySize() {
		this.removeTweet(this.queue.poll());
	}

	protected void removeTweet(List<String> tweet) {
		super.currentHistorySize--;
		for (String word : tweet)
			this.removeWord(word);
	}

	protected void removeWord(String word) {
		super.set.put(word, this.set.get(word) - 1);
		super.totalCount--;
	}

	/**
	 * Clears the current history and queue.
	 * Also calls HistoryRetentionTechnique.reset()
	 */
	@Override
	public void reset() {
		super.reset();
		if (this.queue != null)
			this.queue.clear();
	}
}
