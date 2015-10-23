package moa.classifiers.smoothing.smoothingtechniques.foreground;

import moa.classifiers.smoothing.smoothingtechniques.BackgroundModel;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;

public class AbsoluteDiscounting extends ForegroundModel {

	protected final double delta;

	public AbsoluteDiscounting(final BackgroundModel bm,
							   final HistoryRetentionTechnique history,
							   final double threshold,
							   final double delta) {
		super(bm, history, threshold);
		this.delta = delta;
	}

	@Override
	protected double getProbability(final String word) {
		//max(c(w;h) - delta, 0).
		double topLeft = Math.max(this.history.getWordCount(word) - this.delta, 0);
		// Sum over w, c(w;h).
		double bottom =  this.history.getAllWordsCounts();
		// (delta x w_n) x PBw.
		double topRight = this.delta * this.history.getTotalUniqueWordCount() * this.bm.getProbability(word);

		return (topLeft + topRight) / bottom;
	}
}
