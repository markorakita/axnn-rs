package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

public class MaxPoolLayerCPU extends LayerCPU
{
    /**
     * Width of the pooling unit.
     */
    private final int m_unitWidth;

    /**
     * Height of the pooling unit.
     */
    private final int m_unitHeight;

    /**
     * Padding in dimension X.
     */
    private final int m_paddingX;

    /**
     * Padding in dimension Y.
     */
    private final int m_paddingY;

    /**
     * Stride of the pooling unit.
     */
    private final int m_unitStride;

    public MaxPoolLayerCPU(int inputNumChannels, int inputDataWidth, int inputDataHeight, int unitWidth, int unitHeight,
                           int paddingX, int paddingY, int unitStride)
    {
        m_inputDataNumChannels = inputNumChannels;
        m_inputDataWidth = inputDataWidth;
        m_inputDataHeight = inputDataHeight;
        m_inputDataBufferSize = m_inputDataNumChannels * m_inputDataWidth * m_inputDataHeight;

        m_unitWidth = unitWidth;
        m_unitHeight = unitHeight;
        m_paddingX = paddingX;
        m_paddingY = paddingY;
        m_unitStride = unitStride;

        m_activationDataNumChannels = inputNumChannels;
        m_activationDataWidth = 1 + (int)Math.ceil((double)(m_paddingX + m_inputDataWidth - m_unitWidth) / m_unitStride);
        m_activationDataHeight = 1 + (int)Math.ceil((double)(m_paddingY + m_inputDataHeight - m_unitHeight) / m_unitStride);
        m_activationDataBufferSize = m_activationDataNumChannels * m_activationDataWidth * m_activationDataHeight;

        m_activationDataBuffer = new float[m_activationDataBufferSize];
    }

    @Override
    public void doForwardProp()
    {
        for (int channel = 0; channel < m_inputDataNumChannels; ++channel)
        {
            final int activationChannelOffset = channel * m_activationDataWidth * m_activationDataHeight;
            final int dataChannelOffset = channel * m_inputDataWidth * m_inputDataHeight;
            int startY = -m_paddingY;
            for (int unitY = 0; unitY < m_activationDataHeight; ++unitY)
            {
                int startX = -m_paddingX;
                for (int unitX = 0; unitX < m_activationDataWidth; ++unitX)
                {
                    final int activationDataIndex = activationChannelOffset + unitY * m_activationDataWidth + unitX;
                    m_activationDataBuffer[activationDataIndex] = -Float.MAX_VALUE;
                    for (int currY = startY; currY < startY + m_unitHeight; ++currY)
                    {
                        for (int currX = startX; currX < startX + m_unitWidth; ++currX)
                        {
                            if (currY >= 0 && currY < m_inputDataHeight && currX >= 0 && currX < m_inputDataWidth)
                            {
                                m_activationDataBuffer[activationDataIndex] = Math.max(m_activationDataBuffer[activationDataIndex],
                                        m_inputDataBuffer[dataChannelOffset + currY * m_inputDataWidth + currX]);
                            }
                        }
                    }
                    startX += m_unitStride;
                }
                startY += m_unitStride;
            }
        }
    }
}
