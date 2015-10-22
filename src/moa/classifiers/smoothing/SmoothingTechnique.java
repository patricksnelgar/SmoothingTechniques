package moa.classifiers.smoothing;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.smoothing.smoothingtechniques.*;
import moa.classifiers.smoothing.smoothingtechniques.foreground.*;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.Forget;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.HistoryRetentionTechnique;
import moa.classifiers.smoothing.smoothingtechniques.foreground.history.Queue;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.options.*;
import moa.streams.ArffFileStream;
import weka.core.Instance;

import java.util.*;

public class SmoothingTechnique extends AbstractClassifier implements Classifier {

	//TODO: generate new UID.
	/** For serialization */
	private static final long serialVersionUID = 123456l;

	private static final String Punctuation = "\"'.:-!?,;";

	protected int m_minWordsInTweet = 5;
	protected double m_absoluteDiscountingSigma = 0.9d;
	protected double m_jalinekMercerSmoothingLambda = 0.4d;
	protected double m_bayesianSmoothingMu = 10000d;
	protected double m_stupidBackoffAlpha = 0.3d;
	protected double m_threshold = 200d;
	protected int m_historySize = 1000;
	protected int m_tweetIndex = 0;
	protected String m_hashTag = "";
	protected String m_backgroundDataPath = "";

	protected static final int
			FORGET  = 0,
			QUEUE   = 1;
	public MultiChoiceOption historyRetentionFunctionOption = new MultiChoiceOption(
			"historyRetentionTechnique", 'r', "The history retention function to use.",
			new String[]{ "FORGET", "QUEUE" },
			new String[]{ "Forget", "Queue (FIFO)"},
			QUEUE);

	protected static final int
			ABSOLUTEDISCOUNTING      = 0,
			JALINEKMERCERSMOOTHING   = 1,
			BAYESIANSMOOTHING        = 2,
			STUPIDBACKOFF            = 3;
	public MultiChoiceOption smoothingFunctionOption = new MultiChoiceOption(
			"smoothingTechnique", 'f', "The smoothing function to use.",
			new String[]{ "ABSOLUTEDISCOUNTING", "JALINEKMERCERSMOOTHING", "BAYESIANSMOOTHING", "STUPIDBACKOFF" },
			new String[]{ "Absolute Discounting", "Jalinek-Mercer Smoothing", "Bayesian Smoothing", "Stupid Backoff" },
			STUPIDBACKOFF);

	protected int m_historyTechnique = QUEUE;

	protected int m_foregroundModel = STUPIDBACKOFF;

	public FileOption backgroundDataPathOption = new FileOption("backgroundDataPath",
			'p', "The Background Data Path parameter.",
			"", "arff", false);

	public IntOption minWordsInTweetOption = new IntOption("minWordsInTweet",
			'w', "Min Words in Tweet parameter.",
			10, 0, 70); // Tweets only have 140 characters => 70 char + space.

	public IntOption historySizeOption = new IntOption("historySize",
			'h', "History Size parameter.",
			1000, 0, Integer.MAX_VALUE);

	public FloatOption thresholdOption = new FloatOption("threshold",
			'm', "Threshold parameter.",
			1000f, 0f, Float.MAX_VALUE);


	public IntOption tweetIndexOption = new IntOption("tweetIndex",
			'i', "Tweet Index in data parameter.",
			0, 0, Integer.MAX_VALUE);

	public StringOption hashTagOption = new StringOption("hashTag", 't', "Hash-Tag parameter.", "");

	public FloatOption absoluteDiscountingSigmaOption = new FloatOption("absoluteDiscountingSigma",
			's', "Absolute Discounting Sigma parameter.",
			0.9f, 0f, 1f);

	public FloatOption jalinekMercerSmoothingLambdaOption = new FloatOption("jalinekMercerSmoothingLambda",
			'j', "Jalinek-Mercer Smoothing Lambda parameter.",
			0.4f, 0f, 1f);

	public FloatOption bayesianSmoothingMuOption = new FloatOption("bayesianSmoothingMu",
			'b', "Bayesian Smoothing Mu parameter.",
			10000f, 0f, Float.MAX_VALUE);

	public FloatOption stupidBackoffAlphaOption = new FloatOption("stupidBackoffAlpha",
			'a', "Stupid Backoff Alpha parameter.",
			0.3f, 0f, 1f);

	/**
	 * Set the value of Min Words in Tweet to use.
	 * @param minWordsInTweet the value of Min Words in Tweet to use.
	 */
	public void setMinWordsInTweet(int minWordsInTweet) { m_minWordsInTweet = minWordsInTweet; }

	/**
	 * Get the current value of Min Words in Tweet.
	 * @return the current value of Min Words in Tweet.
	 */
	public int getMinWordsInTweet() { return m_minWordsInTweet; }

	/**
	 * Set the value of Sigma to use in Absolute Discounting.
	 * @param absoluteDiscountingSigma the value of Sigma to use in Absolute Discounting.
	 */
	public void setAbsoluteDiscountingSigma(double absoluteDiscountingSigma) {
		m_absoluteDiscountingSigma = absoluteDiscountingSigma;
	}

	/**
	 * Get the current value of the Absolute Discounting Sigma.
	 * @return the current value of the Absolute Discounting Sigma.
	 */
	public double getAbsoluteDiscountingSigma() { return m_absoluteDiscountingSigma; }

	/**
	 * Set the value of Lambda to use in Jalinek-Mercer Smoothing.
	 * @param jalinekMercerSmoothingLambda the value of Lambda to use in Jalinek-Mercer Smoothing.
	 */
	public void setJalinekMercerSmoothingLambda(double jalinekMercerSmoothingLambda) {
		m_jalinekMercerSmoothingLambda = jalinekMercerSmoothingLambda;
	}

	/**
	 * Get the current value of the Jalinek-Mercer Smoothing Lambda.
	 * @return the current value of the Jalinek-Mercer Smoothing Lambda.
	 */
	public double getJalinekMercerSmoothingLambda() { return m_jalinekMercerSmoothingLambda; }

	/**
	 * Set the value of Mu to use in Bayesian Smoothing.
	 * @param bayesianSmoothingMu the value of Mu to use in Bayesian Smoothing.
	 */
	public void setBayesianSmoothingMu(double bayesianSmoothingMu) {
		m_bayesianSmoothingMu = bayesianSmoothingMu;
	}

	/**
	 * Get the current value of the Bayesian Smoothing Mu.
	 * @return the current value of the JBayesian Smoothing Mu.
	 */
	public double getBayesianSmoothingMu() { return m_bayesianSmoothingMu; }

	/**
	 * Set the value of Alpha to use in Stupid Backoff.
	 * @param stupidBackoffAlpha the value of Alpha to use in Stupid Backoff.
	 */
	public void setStupidBackoffAlpha(double stupidBackoffAlpha) { m_stupidBackoffAlpha = stupidBackoffAlpha; }

	/**
	 * Get the current value of the Stupid Backoff Alpha.
	 * @return the current value of the Stupid Backoff Alpha.
	 */
	public double getStupidBackoffAlpha() { return m_stupidBackoffAlpha; }

	/**
	 * Set the value of history size to use.
	 * @param historySize the value of history size to use.
	 */
	public void setHistorySize(int historySize) { m_historySize = historySize; }

	/**
	 * Get the current value of history size.
	 * @return the current value of history size.
	 */
	public int getHistorySize() { return m_historySize; }

	/**
	 * Set the value of threshold to use.
	 * @param threshold the value of threshold to use.
	 */
	public void setThreshold(double threshold) { m_threshold = threshold; }

	/**
	 * Get the current value of threshold.
	 * @return the current value of threshold.
	 */
	public double getThreshold() { return m_threshold; }

	/**
	 * Get the current value of the tweet index.
	 * @return the current value of the tweet index.
	 */
	public int getTweetIndex() { return m_tweetIndex; }

	/**
	 * Set the value of the tweet index.
	 * @param tweetIndex the value of the tweet index.
	 */
	public void setTweetIndex(int tweetIndex) { m_tweetIndex = tweetIndex; }

	/**
	 * Get the current value of the hash-tag to filter by.
	 * @return the current value of the hash-tag to filter by.
	 */
	public String getBackgroundDataPath() { return m_backgroundDataPath; }

	/**
	 * Set the value of the Background Data path.
	 * @param backgroundDataPath the value of the Background Data path.
	 */
	public void setBackgroundDataPath(String backgroundDataPath) { m_backgroundDataPath = backgroundDataPath; }

	/**
	 * Get the current value of the hash-tag to filter by.
	 * @return the current value of the hash-tag to filter by.
	 */
	public String getHashTag() {
		return m_hashTag;
	}

	/**
	 * Set the value of the hash-tag to filter by.
	 * @param hashTag the value of the hash-tag to filter by.
	 */
	public void setHashTag(String hashTag) {
		if (hashTag != null && hashTag.length() > 0 && !hashTag.startsWith("#"))
			hashTag = "#" + hashTag;	// Ensure that there is a hash-tag on the query tag.
		m_hashTag = hashTag != null ? hashTag.toLowerCase() : "";
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
		if (this.foregroundModel != null) {
			this.foregroundModel.reset();
			this.foregroundModel = null;	// Ensure the background model is rebuilt etc.
		}
	}

	@Override
	public void resetLearningImpl() {
		reset();
		setHistoryTechnique(this.historyRetentionFunctionOption.getChosenIndex());
		setSmoothingTechnique(this.smoothingFunctionOption.getChosenIndex());
		setBackgroundDataPath(this.backgroundDataPathOption.getValue());
		setMinWordsInTweet(this.minWordsInTweetOption.getValue());
		setAbsoluteDiscountingSigma(this.absoluteDiscountingSigmaOption.getValue());
		setJalinekMercerSmoothingLambda(this.jalinekMercerSmoothingLambdaOption.getValue());
		setBayesianSmoothingMu(this.bayesianSmoothingMuOption.getValue());
		setStupidBackoffAlpha(this.stupidBackoffAlphaOption.getValue());
		setHistorySize(this.historySizeOption.getValue());
		setThreshold(this.thresholdOption.getValue());
		setTweetIndex(this.tweetIndexOption.getValue());
		setHashTag(this.hashTagOption.getValue());
	}

	/**
	 * Trains the classifier with the given instance.
	 * @param inst the new training instance to include in the model.
	 */
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (this.foregroundModel == null)
			initializeForegroundModel();

		String instanceTweet =  inst.stringValue(this.getTweetIndex()).toLowerCase();
		List<String> tweet = Arrays.asList(instanceTweet.split(" "));

		/* Check if the tweet conditions are met. */
		if ((tweet = this.filterTweet(tweet, true)) == null)
			return;

		// Update foreground model with new tweet if relevant.
		this.foregroundModel.addTweet(tweet);
	}

	/**
	 * Checks if a tweet is valid.
	 * @param tweet The tweet t
	 * @return Returns the index of the hash-tag or -1 if tweet is invalid.
	 */
	protected List<String> filterTweet(List<String> tweet, boolean isTrain) {
		// Sanitize input.
		boolean changed = false;
		List<String> newTweet = new ArrayList<>();
		for (String word : tweet) {
			if (word.isEmpty())
				continue;
			// Filter leading and trailing punctuation.
			word = word.replaceAll("^[^a-zA-Z#]+", "");

			if (word.isEmpty())
				continue;

			// Skip re-adding the hash-tag.
			if (word.equals(this.getHashTag())) {
				changed = true;
				continue;
			}

			newTweet.add(word);
		}
		tweet = newTweet;

		// Count non-hash-tag words.
		int wordCount = 0;
		for(String w : tweet) {
			if (!w.startsWith("#"))
				wordCount++;
		}

		if (wordCount < this.getMinWordsInTweet())
			return null;

		// If classify, return the tweet cleaned and without topic of interest hash-tag.
		if (!isTrain)
			return tweet;

		// Else is train, check length and that the tweet actually did have the hash-tag.
		if (!changed)   // Hash-tag wasn't in tweet, don't train using tweet.
			return null;

		wordCount = 0;
		// Count non-hash-tag words.
		for(String w : tweet) {
			if (!w.startsWith("#"))
				wordCount++;
		}

		if (wordCount < this.getMinWordsInTweet())
			return null;

		return tweet;
	}

	protected void initializeForegroundModel() {
		BackgroundModel backgroundModel = this.initializeBackgroundModel();

		// Initialize the specified History Retention Technique.
		HistoryRetentionTechnique history;
		switch (this.getHistoryTechnique()) {
			case FORGET : history = new Forget(this.getHistorySize()); break;
			case QUEUE : history = new Queue(this.getHistorySize()); break;
			default : history = new Queue(this.getHistorySize());
		}

		// Initialize the specified foreground model.
		switch (this.getSmoothingTechnique()) {
			case ABSOLUTEDISCOUNTING :
				this.foregroundModel =
						new AbsoluteDiscounting(backgroundModel, history, this.getThreshold(),
								this.getAbsoluteDiscountingSigma());
				break;
			case JALINEKMERCERSMOOTHING :
				this.foregroundModel =
						new JalinekMercerSmoothing(backgroundModel, history, this.getThreshold(),
								this.getJalinekMercerSmoothingLambda());
				break;
			case BAYESIANSMOOTHING :
				this.foregroundModel =
						new BayesianSmoothing(backgroundModel, history, this.getThreshold(),
								this.getBayesianSmoothingMu());
				break;
			case STUPIDBACKOFF :
				this.foregroundModel =
						new StupidBackoff(backgroundModel, history, this.getThreshold(),
								this.getStupidBackoffAlpha());
				break;
			default :
				this.foregroundModel =
						new StupidBackoff(backgroundModel, history, this.getThreshold(),
								this.getStupidBackoffAlpha());
		}
	}

	protected BackgroundModel initializeBackgroundModel() {
		List<String> backgroundWords = new ArrayList<>();
		ArffFileStream stream = new ArffFileStream(this.backgroundDataPathOption.getValue(), -1);
		int tweetIndex = this.getTweetIndex();
		while (stream.hasMoreInstances()) {
			String tweet = stream.nextInstance().stringValue(tweetIndex).toLowerCase();
			List<String> tweetWords = Arrays.asList(tweet.split(" "));
			// Invalid tweet.
			if ((tweetWords = this.filterTweet(tweetWords, true)) == null)
				continue;

			backgroundWords.addAll(tweetWords);
		}

		BackgroundModel backgroundModel = new BackgroundModel();
		backgroundModel.buildProbabilities(backgroundWords);
		return backgroundModel;
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 * @param inst the instance to be classified.
	 * @return predicted class probability distribution.
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {
		if (this.foregroundModel == null)
			initializeForegroundModel();

		String instanceTweet =  inst.stringValue(this.getTweetIndex()).toLowerCase();
		List<String> tweet = new ArrayList<>(Arrays.asList(instanceTweet.split(" ")));

		/* Check if the tweet conditions are met. */
		tweet = this.filterTweet(tweet, false);
		if (tweet == null)
			return new double[0];

		// If 1 predicts (1 - 1, 1) (0, 1) (so class 1) else predicts (1 - 0,0) (1, 0) (so class 0)
		double classification = this.foregroundModel.getClassification(tweet) ? 1d : 0d;

		/* Precision Recall */
		if (classification == 1)
			if (inst.classValue() == 1)
				this.truePositive++;
			else
				this.falsePositive++;
		else
			if (inst.classValue() == 1)
				this.falseNegative++;
			else
				this.trueNegative++;

		return new double[] { 1 - classification, classification };
	}

	/**
	 * Prints out the classifier.
	 * @return a description of the classifier as a string.
	 */
	//TODO:
	public String toString() { return "todo"; }

	double truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		final double tpfp = this.truePositive + this.falsePositive,
				tpfn = this.truePositive + this.falseNegative,
				totalSeen = this.trueNegative + this.truePositive + this.falseNegative + this.falsePositive;

		double precision = tpfp == 0d ? 0d : this.truePositive / tpfp;
		double recall = tpfn == 0d ? 0d : this.truePositive / tpfn;

		double accuracy = totalSeen == 0d ? 0d : (this.trueNegative + this.truePositive) / totalSeen;

		return new Measurement[]{
				new Measurement("Precision", precision),
				new Measurement("Recall", recall),
				new Measurement("Accuracy", accuracy)
		};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		StringUtils.appendIndented(out, indent, toString());
		StringUtils.appendNewline(out);
	}

	@Override
	public boolean isRandomizable() { return false; }
}
