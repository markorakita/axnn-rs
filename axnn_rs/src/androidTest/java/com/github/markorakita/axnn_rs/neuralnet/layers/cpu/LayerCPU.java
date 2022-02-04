package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

public abstract class LayerCPU
{
    /**
     * Number of input data channels.
     */
    protected int m_inputDataNumChannels;

    /**
     * Input data width.
     */
    protected int m_inputDataWidth;

    /**
     * Input data height.
     */
    protected int m_inputDataHeight;

    /**
     * Input data buffer.
     *
     * CPU layers have per-channel data structure (R1R2G1G2B1B2).
     */
    protected float[] m_inputDataBuffer;

    /**
     * Input data buffer size.
     */
    protected int m_inputDataBufferSize;

    /**
     * Number of activation data channels.
     */
    protected int m_activationDataNumChannels;

    /**
     * Activation data width.
     */
    protected int m_activationDataWidth;

    /**
     * Activation data height.
     */
    protected int m_activationDataHeight;

    /**
     * Activations buffer.
     */
    protected float[] m_activationDataBuffer;

    /**
     * Activations buffer size.
     */
    protected int m_activationDataBufferSize;

    /**
     * Sets input data buffer.
     * @param inputDataBuffer Input data buffer to set.
     */
    public void setInputDataBuffer(float[] inputDataBuffer)
    {
        m_inputDataBuffer = inputDataBuffer;
    }

    /**
     * Loads inputs from input data buffer structured per-pixel (R1G1B1R2G2B2).
     */
    public void setInputDataFromPerPixelBuffer(float[] inputDataBufferPP)
    {
        m_inputDataBuffer = new float[m_inputDataBufferSize];
        final int inputDataSize = m_inputDataWidth * m_inputDataHeight;
        for (int channel = 0; channel < m_inputDataNumChannels; ++channel)
        {
            final int inputOffset = channel * inputDataSize;
            for (int px = 0; px < inputDataSize; ++px)
            {
                m_inputDataBuffer[inputOffset + px] = inputDataBufferPP[px * m_inputDataNumChannels + channel];
            }
        }
    }

    /**
     * Gets number of activation channels.
     */
    public int getActivationDataNumChannels()
    {
        return m_activationDataNumChannels;
    }

    /**
     * Gets activation data width.
     */
    public int getActivationDataWidth()
    {
        return m_activationDataWidth;
    }

    /**
     * Gets activation data height.
     */
    public int getActivationDataHeight()
    {
        return m_activationDataHeight;
    }

    /**
     * Gets activation data buffer.
     */
    public float[] getActivationDataBuffer()
    {
        return m_activationDataBuffer;
    }

    /**
     * Gets activations data buffer with per-pixel structure (R1G1B1R2G2B2).
     */
    public float[] getActivationDataPerPixelBuffer()
    {
        float[] activationDataBufferPP = new float[m_activationDataBufferSize];
        final int activationDataSize = m_activationDataWidth * m_activationDataHeight;
        for (int px = 0; px < activationDataSize; ++px)
        {
            final int activationOffset = px * m_activationDataNumChannels;
            for (int channel = 0; channel < m_activationDataNumChannels; ++channel)
            {
                activationDataBufferPP[activationOffset + channel] = m_activationDataBuffer[channel * activationDataSize + px];
            }
        }

        return activationDataBufferPP;
    }

    /**
     * Does forward propagation through layer.
     */
    public abstract void doForwardProp();
}
