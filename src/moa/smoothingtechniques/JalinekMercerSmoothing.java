package moa.smoothingtechniques;

public class JalinekMercerSmoothing extends ForegroundModel {

	protected final double lambda, invLambda;

	public JalinekMercerSmoothing(final BackgroundModel bm,
								  final HistoryRetentionTechnique history,
								  final double lambda) {
		super(bm, history);
		this.lambda = lambda;
		this.invLambda = 1 - lambda;
	}

	@Override
	protected double getProbability(final String word) {
		// lambda x (c(w;h) / Sum c(w;h)) + (1 - lambda) x P_B(w)
		return (this.lambda * (this.history.getWordCount(word) / this.history.getTotalCount())) +
				(this.invLambda * this.bm.getProbability(word));
	}
}
