package moa.smoothingtechniques;

public class BayesianSmoothing extends ForegroundModel {

	protected final double mu;

	public BayesianSmoothing(final BackgroundModel bm, final HistoryRetentionTechnique history, final double mu) {
		super(bm, history);
		this.mu = mu;
	}

	@Override
	protected double getProbability(final String word) {
		// c(w;h) + mu x P_B(W) / (Sum over w c(w;h)) + mu
		return (this.history.getWordCount(word) + (this.mu * this.bm.getProbability(word))) /
				(this.history.getTotalCount() + this.mu);
	}
}
