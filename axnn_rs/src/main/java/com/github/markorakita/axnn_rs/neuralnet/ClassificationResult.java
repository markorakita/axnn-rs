package com.github.markorakita.axnn_rs.neuralnet;

/**
 * Result of the classification.
 */
public class ClassificationResult
{
    /**
     * Predicted class (starting from zero).
     */
    private final int m_predictedClass;

    /**
     * Probability that predicted class is correct.
     */
    private final float m_predictionProbability;

    /**
     * Constructor.
     * @param predictedClass Predicted class.
     * @param predictionProbability Prediction probability.
     */
    public ClassificationResult(int predictedClass, float predictionProbability)
    {
        m_predictedClass = predictedClass;
        m_predictionProbability = predictionProbability;
    }

    /**
     * Returns predicted class (starting from zero).
     */
    public int getPredictedClass()
    {
        return m_predictedClass;
    }

    /**
     * Returns prediction probability.
     */
    public float getPredictionProbability()
    {
        return m_predictionProbability;
    }
}
