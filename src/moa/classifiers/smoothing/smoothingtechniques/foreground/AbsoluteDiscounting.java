package moa.classifiers.smoothing.smoothingtechniques.foreground;

import moa.classifiers.smoothing.smoothingtechniques.BackgroundModel;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;

public class AbsoluteDiscounting extends ForegroundModel {

	protected final double sigma;

	public AbsoluteDiscounting(final BackgroundModel bm,
							   final HistoryRetentionTechnique history,
							   final double threshold,
							   final double sigma) {
		super(bm, history, threshold);
		this.sigma = sigma;
	}

	@Override
	protected double getProbability(final String word) {
		//max(c(w;h) - sigma, 0).
		double topLeft = Math.max(this.history.getWordCount(word) - this.sigma, 0);
		// Sum over w, c(w;h).
		double bottom =  this.history.getAllWordsCounts();
		// (sigma x w_n) x PBw.
		double topRight = this.sigma * this.history.getTotalUniqueWordCount() * this.bm.getProbability(word);

		return (topLeft + topRight) / bottom;
	}
}
