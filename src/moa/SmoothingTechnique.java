package moa;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.core.StringUtils;
import moa.options.*;
import moa.smoothingtechniques.*;
import moa.smoothingtechniques.Queue;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.*;

public class SmoothingTechnique extends AbstractClassifier implements Regressor {

	//TODO: generate new UID.
	/** For serialization */
	private static final long serialVersionUID = 123456l;

	private static final Set<String> StopWords = new HashSet<>(Arrays.asList(
			new String[] { "I", "a", "about", "an", "are", "as", "at", "be", "by", "com",
					"for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
					"when", "where", "who", "will", "with", "the", "www" }
	));

	protected int m_minWordsInTweet = 5;
	protected boolean m_doStopWordChecking = true;
	protected double m_absoluteDiscountingSigma = 0.9;
	protected double m_jalinekMercerSmoothingLambda = 0.4;
	protected double m_bayesianSmoothingMu = 10000;
	protected double m_stupidBackoffAlpha = 0.3;
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

	public FlagOption doStopWordChecking = new FlagOption("doStopWordChecking",
			'c', "Stop words don't add to the Min Words in Tweet count.");

	public IntOption historySizeOption = new IntOption("historySize",
			'h', "History Size parameter.",
			1000, 0, Integer.MAX_VALUE);

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
	 * Set the value of Do Stop Word Checking.
	 * @param doStopWordChecking the value of Do Stop Word Checking.
	 */
	public void setDoStopWordChecking(boolean doStopWordChecking) { m_doStopWordChecking = doStopWordChecking; }

	/**
	 * Get the current value of Do Stop Word Checking.
	 * @return the current value of Do Stop Word Checking.
	 */
	public boolean getDoStopWordChecking() { return m_doStopWordChecking; }

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
		m_hashTag = hashTag;
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
		setDoStopWordChecking(this.doStopWordChecking.isSet());
		setAbsoluteDiscountingSigma(this.absoluteDiscountingSigmaOption.getValue());
		setJalinekMercerSmoothingLambda(this.jalinekMercerSmoothingLambdaOption.getValue());
		setBayesianSmoothingMu(this.bayesianSmoothingMuOption.getValue());
		setStupidBackoffAlpha(this.stupidBackoffAlphaOption.getValue());
		setHistorySize(this.historySizeOption.getValue());
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
		List<String> tweet = new ArrayList<>(Arrays.asList(instanceTweet.split(" ")));

		/* Check if the tweet conditions are met. */
		int index = this.filterTweet(tweet);
		if (index < 0) return;
		tweet.remove(index);

		/* Intrinsic Evaluation - Perplexity (Called in this.getVotesForInstance()) */
		//double perplexity = this.foregroundModel.getPerplexity(tweet);

		// Update foreground model with new tweet.
		this.foregroundModel.addTweet(tweet);
	}

	/**
	 * Checks if a tweet is valid.
	 * @param tweet The tweet t
	 * @return Returns the index of the hash-tag or -1 if tweet is invalid.
	 */
	protected int filterTweet(List<String> tweet) {
		int wordCount = 0;
		if (this.getDoStopWordChecking()) {
			for (String word : tweet) {
				if (!SmoothingTechnique.StopWords.contains(word))
					wordCount++;
			}
		}
		else {
			wordCount = tweet.size();
		}

		if (wordCount < this.getMinWordsInTweet() + 1)
			return -1;

		return tweet.indexOf(this.getHashTag());
	}

	protected void initializeForegroundModel() {
		File backgroundFile = this.backgroundDataPathOption.getFile();

		BackgroundModel backgroundModel = null;
		try {
			backgroundModel = this.initializeBackgroundModel(backgroundFile, this.getTweetIndex());
		} catch (IOException e) {
			e.printStackTrace();
		}

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
						new AbsoluteDiscounting(backgroundModel, history, this.getAbsoluteDiscountingSigma());
				break;
			case JALINEKMERCERSMOOTHING :
				this.foregroundModel =
						new JalinekMercerSmoothing(backgroundModel, history, this.getJalinekMercerSmoothingLambda());
				break;
			case BAYESIANSMOOTHING :
				this.foregroundModel =
						new BayesianSmoothing(backgroundModel, history, this.getBayesianSmoothingMu());
				break;
			case STUPIDBACKOFF :
				this.foregroundModel =
						new StupidBackoff(backgroundModel, history, this.getStupidBackoffAlpha());
				break;
			default :
				this.foregroundModel =
						new StupidBackoff(backgroundModel, history, this.getStupidBackoffAlpha());
		}
	}

	protected BackgroundModel initializeBackgroundModel(File file, int tweetIndex) throws IOException {
		List<String> backgroundWords = new ArrayList<>();
		BufferedReader reader = null;

		try {
			reader = new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		if (reader == null)
			throw new IOException("File not found or able to be opened.");

		try {
			ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
			Instances data = arff.getStructure();
			Instance param;
			while ((param = arff.readInstance(data)) != null) {
				String tweet = param.stringValue(tweetIndex);
				List<String> tweetWords = new ArrayList<>(Arrays.asList(tweet.split(" ")));
				int index = this.filterTweet(tweetWords);
				// Invalid tweet.
				if (index < 0) continue;
				tweetWords.remove(index);

				backgroundWords.addAll(tweetWords);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			reader.close();
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
	//TODO:
	@Override
	public double[] getVotesForInstance(Instance inst) {
		if (this.foregroundModel == null)
			initializeForegroundModel();

		String instanceTweet =  inst.stringValue(this.getTweetIndex()).toLowerCase();
		List<String> tweet = new ArrayList<>(Arrays.asList(instanceTweet.split(" ")));

		/* Check if the tweet conditions are met. */
		int index = this.filterTweet(tweet);
		if (index < 0) return new double[] { 0 };
		tweet.remove(index);

		/* Intrinsic Evaluation - Perplexity */
		return new double[] {this.foregroundModel.getPerplexity(tweet)};
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
