package moa;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.smoothingtechniques.*;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SmoothingTechnique extends AbstractClassifier implements Regressor {

	//TODO: generate new UID.
	/** For serialization */
	private static final long serialVersionUID = 123456l;
	private static final int MinNumberOfWords = 10;

	protected double m_alpha = 0.3;
	protected int m_historySize = 1000;
	protected String m_hashTag = "#todo";

	public FloatOption alphaOption = new FloatOption("alpha",
			'a', "Alpha parameter.",
			0.0001f, 0f, Float.MAX_VALUE);

	public IntOption historySizeOption = new IntOption("historySize",
			'h', "History Size parameter.",
			1000, 0, Integer.MAX_VALUE);

	protected static final int FORGET = 0, QUEUE = 1;
	public MultiChoiceOption historyRetentionFunctionOption = new MultiChoiceOption(
			"historyRetentionTechnique", 'f', "The history retention function to use.", new String[]{
				"FORGET", "QUEUE" }, new String[]{
				"Forget",
				"Queue (FIFO)"}, 1);

	protected static final int STUPIDBACKOFF = 0;
	public MultiChoiceOption smoothingFunctionOption = new MultiChoiceOption(
			"smoothingTechnique", 's', "The smoothing function to use.", new String[]{
				"STUPIDBACKOFF" }, new String[]{
				"Stupid Backoff" }, 0);

	protected int m_historyTechnique = QUEUE;

	protected int m_foregroundModel = STUPIDBACKOFF;



	/**
	 * Set the value of alpha to use.
	 * @param alpha the value of alpha to use.
	 */
	public void setAlpha(double alpha) {
		m_alpha = alpha;
	}

	/**
	 * Get the current value of alpha.
	 * @return the current value of alpha.
	 */
	public double getAlpha() {
		return m_alpha;
	}

	/**
	 * Set the value of history size to use.
	 * @param historySize the value of history size to use.
	 */
	public void setHistorySize(int historySize) {
		m_historySize = historySize;
	}

	/**
	 * Get the current value of history size.
	 * @return the current value of history size.
	 */
	public int getHistorySize() {
		return m_historySize;
	}

	/**
	 * Set the History Technique to use.
	 * @param Technique the History Technique to use.
	 */
	public void setHistoryTechnique(int Technique) { m_historyTechnique = Technique;  }

	/**
	 * Get the current History Technique.
	 * @return the current History Technique.
	 */
	public int getHistoryTechnique() { return m_historyTechnique; }

	/**
	 * Set the Smoothing Technique to use.
	 * @param Technique the Smoothing Technique to use.
	 */
	public void setSmoothingTechnique(int Technique) { m_foregroundModel = Technique; }

	/**
	 * Get the current Smoothing Technique.
	 * @return the current Smoothing Technique.
	 */
	public int getSmoothingTechnique() { return m_foregroundModel; }

	protected ForegroundModel foregroundModel = null;

	public void reset() {
		this.foregroundModel.reset();
	}

	@Override
	public void resetLearningImpl() {
		reset();
		setAlpha(this.alphaOption.getValue());
		setHistorySize(this.historySizeOption.getValue());
		setHistoryTechnique(this.historyRetentionFunctionOption.getChosenIndex());
		setSmoothingTechnique(this.smoothingFunctionOption.getChosenIndex());
	}

	/**
	 * Trains the classifier with the given instance.
	 * @param inst the new training instance to include in the model.
	 */
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		/* Initialize the Foreground Model (including the Background Model and the History Retention Technique). */

		if (this.foregroundModel == null) {
			// Initialize the Background Model.
			// TODO: get a list of words from background instances.
			List<String> backgroundWords = new ArrayList<>();

			List<String> backgroundTweets = new ArrayList<>();
			for (String t : backgroundTweets) {
				List<String> tweet = new ArrayList<>(Arrays.asList(t.split(" ")));
				int index = this.filterTweet(tweet);
				// Invalid tweet.
				if (index < 0) continue;
				tweet.remove(index);

				backgroundTweets.addAll(tweet);
			}

			BackgroundModel backgroundModel = new BackgroundModel();
			backgroundModel.buildProbabilities(backgroundWords);

			// Initialize the specified History Retention Technique.
			HistoryRetentionTechnique history;
			switch (this.getHistoryTechnique()) {
				case FORGET : history = new Forget(this.getHistorySize()); break;
				case QUEUE : history = new Queue(this.getHistorySize()); break;
				default : history = new Queue(this.getHistorySize());
			}

			// Initialize the specified foreground model.
			switch (this.getSmoothingTechnique()) {
				case STUPIDBACKOFF : this.foregroundModel = new StupidBackoff(backgroundModel, history, this.getAlpha()); break;
				default : this.foregroundModel = new StupidBackoff(backgroundModel, history, this.getAlpha());
			}
		}

		// TODO: Get words from instance (without the relevant hash-tag).
		String instanceTweet = "string from instance".toLowerCase();

		List<String> tweet = new ArrayList<>(Arrays.asList(instanceTweet.split(" ")));


		/* Check if the tweet conditions are met. */

		// Check and remove the hash-tag from the tweet.
		int index = this.filterTweet(tweet);
		// Invalid tweet.
		if (index < 0)
			return;
		tweet.remove(index);

		/* Process tweet. */

		// Intrinsic Evaluation - Perplexity (before adding the new instance)).
		double perplexity = this.foregroundModel.getPerplexity(tweet);

		// Update foreground model with new tweet.
		this.foregroundModel.addTweet(tweet);
	}

	/**
	 * Checks if a tweet is valid.
	 * @param tweet The tweet t
	 * @return Returns the index of the hash-tag or -1 if tweet is invalid.
	 */
	protected int filterTweet(List<String> tweet) {
		// TODO: Add in not counting stop words, and setting min # of words.
		// Check tweet length isn't too small, skip the tweet if it is.
		if (tweet.size() < SmoothingTechnique.MinNumberOfWords)
			return -1;

		return tweet.indexOf(this.m_hashTag);
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 * @param inst the instance to be classified.
	 * @return predicted class probability distribution.
	 */
    //TODO:
	@Override
	public double[] getVotesForInstance(Instance inst) {
		return new double[0];
	}

	/**
	 * Prints out the classifier.
	 * @return a description of the classifier as a string.
	 */
    //TODO:
	public String toString() { return "todo"; }

	@Override
	protected Measurement[] getModelMeasurementsImpl() { return null; }

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		StringUtils.appendIndented(out, indent, toString());
		StringUtils.appendNewline(out);
	}

	@Override
	public boolean isRandomizable() { return false; }
}
