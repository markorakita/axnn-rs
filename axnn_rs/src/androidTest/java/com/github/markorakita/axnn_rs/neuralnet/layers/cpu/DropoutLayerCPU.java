package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

public class DropoutLayerCPU extends LayerCPU
{
    /**
     * Probability to drop some neuron activation.
     */
    private final float m_dropProbability;

    public DropoutLayerCPU(int inputNumChannels, int inputDataWidth, int inputDataHeight, float dropProbability)
    {
        m_inputDataNumChannels = m_activationDataNumChannels = inputNumChannels;
        m_inputDataWidth = m_activationDataWidth = inputDataWidth;
        m_inputDataHeight = m_activationDataHeight = inputDataHeight;
        m_inputDataBufferSize = m_activationDataBufferSize = inputNumChannels * inputDataWidth * inputDataHeight;

        m_dropProbability = dropProbability;

        m_activationDataBuffer = new float[m_activationDataBufferSize];
    }

    @Override
    public void doForwardProp()
    {
        for (int i = 0; i < m_inputDataBuffer.length; ++i)
        {
            m_activationDataBuffer[i] = m_inputDataBuffer[i] * (1.f - m_dropProbability);
        }
    }
}
