package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

public class ResponseNormalizationLayerCPU extends LayerCPU
{
    /**
     * Depth of normalization.
     */
    private final int m_depth;

    /**
     * Normalization bias.
     */
    private final float m_bias;

    /**
     * Normalization alpha coefficient (see the formula in class header).
     */
    private final float m_alphaCoeff;

    /**
     * Normalization beta coefficient (see the formula in class header).
     */
    private final float m_betaCoeff;

    public ResponseNormalizationLayerCPU(int inputNumChannels, int inputDataWidth, int inputDataHeight, int depth, float bias,
                                         float alphaCoeff, float betaCoeff)
    {
        m_inputDataNumChannels = inputNumChannels;
        m_inputDataWidth = inputDataWidth;
        m_inputDataHeight = inputDataHeight;
        m_inputDataBufferSize = m_inputDataNumChannels * m_inputDataWidth * m_inputDataHeight;

        m_depth = depth;
        m_bias = bias;
        // Adjusting alpha coefficient upfront, according to formula.
        m_alphaCoeff = alphaCoeff / depth;
        m_betaCoeff = betaCoeff;

        m_activationDataNumChannels = inputNumChannels;
        m_activationDataWidth = inputDataWidth;
        m_activationDataHeight = inputDataHeight;
        m_activationDataBufferSize = m_inputDataNumChannels * m_activationDataWidth * m_activationDataHeight;

        m_activationDataBuffer = new float[m_activationDataBufferSize];
    }

    @Override
    public void doForwardProp()
    {
        final int inputDataSize = m_inputDataWidth * m_inputDataHeight;

        for (int pixelY = 0; pixelY < m_inputDataHeight; ++pixelY)
        {
            final int pixelOffsetY = pixelY * m_inputDataWidth;
            for (int pixelX = 0; pixelX < m_inputDataWidth; ++pixelX)
            {
                final int pixelOffset = pixelOffsetY + pixelX;
                float crossChannelSum = 0.f;
                for (int channel = 0; channel < m_inputDataNumChannels; ++channel)
                {
                    final int bufferOffset = channel * inputDataSize + pixelOffset;
                    if (channel == 0)
                    {
                        final int actualStartChannel = channel - m_depth / 2;
                        final int startChannel = Math.max(actualStartChannel, 0);
                        final int endChannel = Math.min(actualStartChannel + m_depth, m_inputDataNumChannels);
                        for (int currChannel = startChannel; currChannel < endChannel; ++currChannel)
                        {
                            final float channelData = m_inputDataBuffer[currChannel * inputDataSize + pixelOffset];
                            crossChannelSum += channelData * channelData;
                        }
                    }
                    else
                    {
                        final int channelToSubtract = channel - m_depth / 2 - 1;
                        final int channelToAdd = channelToSubtract + m_depth;
                        if (channelToSubtract >= 0)
                        {
                            final float channelToSubtractData = m_inputDataBuffer[channelToSubtract * inputDataSize + pixelOffset];
                            crossChannelSum -= channelToSubtractData * channelToSubtractData;
                        }
                        if (channelToAdd < m_inputDataNumChannels)
                        {
                            final float channelToAddData = m_inputDataBuffer[channelToAdd * inputDataSize + pixelOffset];
                            crossChannelSum += channelToAddData * channelToAddData;
                        }
                    }

                    m_activationDataBuffer[bufferOffset] = m_inputDataBuffer[bufferOffset] *
                                                             (float)Math.pow(m_bias + m_alphaCoeff * crossChannelSum, -m_betaCoeff);
                }
            }
        }
    }
}
