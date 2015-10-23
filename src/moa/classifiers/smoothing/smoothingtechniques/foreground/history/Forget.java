package moa.classifiers.smoothing.smoothingtechniques.foreground.history;

import java.util.List;

public class Forget extends HistoryRetentionTechnique {

	public Forget(final int historySize) {
		super(historySize);
	}

	protected void addInternalTweet(final List<String> tweet) {
		for (String word : tweet)
			this.addWord(word);
	}

	/**
	 * Clears all history when the max history is reached.
	 */
	@Override
	protected void handleMaxHistorySize() {
		super.reset();
		super.currentHistorySize = 0;
	}
}
