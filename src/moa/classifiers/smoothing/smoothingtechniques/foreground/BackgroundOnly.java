package moa.classifiers.smoothing.smoothingtechniques.foreground;

import moa.classifiers.smoothing.smoothingtechniques.BackgroundModel;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;

public class BackgroundOnly extends ForegroundModel {

	public BackgroundOnly(final BackgroundModel bm,
						  final HistoryRetentionTechnique history,
						  final double threshold) {
		super(bm, history, threshold);
	}

	@Override
	protected double getProbability(final String word) {
		return super.bm.getProbability(word);
	}
}
