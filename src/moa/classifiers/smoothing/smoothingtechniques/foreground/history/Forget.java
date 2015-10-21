package moa.classifiers.smoothing.smoothingtechniques.foreground.history;

public class Forget extends HistoryRetentionTechnique {

	public Forget(final int historySize) {
		super(historySize);
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
