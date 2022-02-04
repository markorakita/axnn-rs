package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

import java.util.Random;

public class StandardLayerCPU extends LayerCPU
{
    private float[] m_weightsBuffer;

    private final int m_weightsBufferSize;

    private float[] m_biasesBuffer;

    private final int m_biasesBufferSize;

    public StandardLayerCPU(int inputNumChannels, int inputDataWidth, int inputDataHeight, int numNeurons)
    {
        m_inputDataBufferSize = inputNumChannels * inputDataWidth * inputDataHeight;
        m_activationDataBufferSize = numNeurons;

        m_weightsBufferSize = numNeurons * m_inputDataBufferSize;
        m_biasesBufferSize = numNeurons;

        allocateBuffers();
    }

    private void allocateBuffers()
    {
        m_weightsBuffer = new float[m_weightsBufferSize];
        Random rand = new Random();
        for (int i = 0; i < m_weightsBufferSize; ++i)
        {
            m_weightsBuffer[i] = 0.01f * (float)rand.nextGaussian();
        }

        m_biasesBuffer = new float[m_biasesBufferSize];
        for (int i = 0; i < m_biasesBufferSize; ++i)
        {
            m_biasesBuffer[i] = 1.0f;
        }

        m_activationDataBuffer = new float[m_activationDataBufferSize];
    }

    public float[] getWeightsBuffer()
    {
        return m_weightsBuffer;
    }

    public float[] getBiasesBuffer()
    {
        return m_biasesBuffer;
    }

    @Override
    public void doForwardProp()
    {
        for (int neuronIndex = 0; neuronIndex < m_activationDataBufferSize; ++neuronIndex)
        {
            final int weightsOffset = neuronIndex * m_inputDataBufferSize;
            m_activationDataBuffer[neuronIndex] = 0.0f;
            for (int inputIndex = 0; inputIndex < m_inputDataBufferSize; ++inputIndex)
            {
                m_activationDataBuffer[neuronIndex] += m_weightsBuffer[weightsOffset + inputIndex] * m_inputDataBuffer[inputIndex];
            }

            // Ignoring activation function here since activation functions are tested separately.
            m_activationDataBuffer[neuronIndex] += m_biasesBuffer[neuronIndex];
        }
    }
}
