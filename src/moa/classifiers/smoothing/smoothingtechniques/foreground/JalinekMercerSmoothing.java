package moa.classifiers.smoothing.smoothingtechniques.foreground;

import moa.classifiers.smoothing.smoothingtechniques.BackgroundModel;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;

public class JalinekMercerSmoothing extends ForegroundModel {

	protected final double lambda, invLambda;

	public JalinekMercerSmoothing(final BackgroundModel bm,
								  final HistoryRetentionTechnique history,
								  final double threshold,
								  final double lambda) {
		super(bm, history, threshold);
		this.lambda = lambda;
		this.invLambda = 1 - lambda;
	}

	@Override
	protected double getProbability(final String word) {
		int count = this.history.getAllWordsCounts();
		double backgroundInfluence = this.invLambda * this.bm.getProbability(word);
		// Remove divide by zero error (assume div by 0 = 0).
		if (count == 0)
			return backgroundInfluence;

		double foregroundInfluence = this.lambda * (this.history.getWordCount(word) / count);

		// lambda x (c(w;h) / Sum c(w;h)) + (1 - lambda) x P_B(w)
		return foregroundInfluence + backgroundInfluence;
	}
}
