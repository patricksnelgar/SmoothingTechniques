package moa.smoothingtechniques;

public class AbsoluteDiscounting extends ForegroundModel {

	protected final double sigma;

	public AbsoluteDiscounting(final BackgroundModel bm, final HistoryRetentionTechnique history, final double sigma) {
		super(bm, history);
		this.sigma = sigma;
	}

	@Override
	protected double getProbability(final String word) {
		return (Math.max(this.history.getWordCount(word) - this.sigma, 0) / this.history.getTotalCount()) +
				((this.sigma * this.history.getTotalUniqueWordCount()) / this.history.getTotalCount());
	}
}
