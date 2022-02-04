package com.github.markorakita.axnn_rs.neuralnet.layers;

/**
 * Output layer, outputs predicted class and prediction probabilities.
 */
public class OutputLayerRS extends LayerRS
{
    /**
     * Supported loss function types.
     */
    public enum LossFunctionType
    {
        LogisticRegression,

        // With Cross Entropy loss we always expect that SoftMax layer is added to the network previous to this Output layer.
        CrossEntropy;
    }

    private static final float c_defaultClassificationThreshold = 0.5f;

    private final LossFunctionType m_lossFunctionType;

    private final float m_classificationThreshold;

    private final float[] m_hostInputDataBuffer;

    private int m_predictedClass;

    private float m_predictionProbability;

    /**
     * Constructor.
     * @param inputDataSize Size of the input data buffer.
     * @param lossFunctionType Loss function type.
     */
    public OutputLayerRS(int inputDataSize, LossFunctionType lossFunctionType)
    {
        this(inputDataSize, lossFunctionType, 0.f);
    }

    /**
     * Constructor.
     * @param inputDataSize Size of the input data buffer.
     * @param lossFunctionType Loss function type.
     * @param classificationThreshold Classification threshold to use.
     */
    public OutputLayerRS(int inputDataSize, LossFunctionType lossFunctionType, float classificationThreshold)
    {
        m_inputDataBufferSize = m_activationDataBufferSize = inputDataSize;

        m_lossFunctionType = lossFunctionType;
        m_classificationThreshold = classificationThreshold > 0.f ? classificationThreshold : c_defaultClassificationThreshold;

        m_hostInputDataBuffer = new float[m_inputDataBufferSize];
    }

    /**
     * Gets predicted class.
     */
    public int getPredictedClass()
    {
        return m_predictedClass;
    }

    /**
     * Gets prediction probability.
     */
    public float getPredictionProbability()
    {
        return m_predictionProbability;
    }

    /**
     * Does forward propagation through layer.
     */
    @Override
    public void doForwardProp()
    {
        m_inputDataBuffer.copyTo(m_hostInputDataBuffer);

        if (m_lossFunctionType == LossFunctionType.LogisticRegression)
        {
            calculateLogisticRegressionStatistics();
        }
        else if (m_lossFunctionType == LossFunctionType.CrossEntropy)
        {
            calculateCrossEntropyStatistics();
        }
    }

    private void calculateLogisticRegressionStatistics()
    {
        float sigmoidActivation = calculateSigmoidActivation();

        if (sigmoidActivation < m_classificationThreshold)
        {
            m_predictedClass = 0;
            m_predictionProbability = 1.0f - sigmoidActivation;
        }
        else
        {
            m_predictedClass = 1;
            m_predictionProbability = sigmoidActivation;
        }
    }

    private float calculateSigmoidActivation()
    {
        return (float)(m_hostInputDataBuffer[0] >= 0.f ?
                (1.0 / (1.0 + Math.exp(-m_hostInputDataBuffer[0]))) :
                (1.0 - 1.0 / (1.0 + Math.exp(m_hostInputDataBuffer[0]))));
    }

    private void calculateCrossEntropyStatistics()
    {
        // With Cross Entropy loss we always expect that SoftMax layer is previous to this Output layer,
        // so that inputs to output layer are soft-max probabilities.
        m_predictedClass = 0;
        m_predictionProbability = m_hostInputDataBuffer[0];
        for (int i = 1; i < m_inputDataBufferSize; ++i)
        {
            if (m_hostInputDataBuffer[i] > m_hostInputDataBuffer[m_predictedClass])
            {
                m_predictedClass = i;
                m_predictionProbability = m_hostInputDataBuffer[i];
            }
        }
    }
}
