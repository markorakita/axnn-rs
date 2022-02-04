package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

import java.util.Random;

public class ConvolutionalLayerCPU extends LayerCPU
{
    private final int m_numFilters;

    private final int m_filterWidth;

    private final int m_filterHeight;

    private final int m_filterSize;

    private float[] m_filtersBuffer;

    private final int m_filtersBufferSize;

    private float[] m_biasesBuffer;

    private final int m_biasesBufferSize;

    private final int m_paddingX;

    private final int m_paddingY;

    private final int m_stride;

    private final int m_numPatchesX;

    private final int m_numPatchesY;

    public ConvolutionalLayerCPU(int inputNumChannels, int inputDataWidth, int inputDataHeight, int numFilters, int filterWidth, int filterHeight,
                                 int paddingX, int paddingY, int stride)
    {
        m_inputDataNumChannels = inputNumChannels;
        m_inputDataWidth = inputDataWidth;
        m_inputDataHeight = inputDataHeight;
        m_inputDataBufferSize = inputNumChannels * inputDataWidth * inputDataHeight;

        m_numFilters = numFilters;
        m_filterWidth = filterWidth;
        m_filterHeight = filterHeight;
        m_filterSize = m_filterWidth * m_filterHeight;
        m_filtersBufferSize = m_numFilters * m_filterSize * m_inputDataNumChannels;
        m_biasesBufferSize = numFilters;

        m_paddingX = paddingX;
        m_paddingY = paddingY;
        m_stride = stride;
        m_numPatchesX = 1 + (int)Math.ceil((double)(2 * paddingX + m_inputDataWidth - m_filterWidth) / m_stride);
        m_numPatchesY = 1 + (int)Math.ceil((double)(2 * paddingY + m_inputDataHeight - m_filterHeight) / m_stride);

        m_activationDataNumChannels = m_numFilters;
        m_activationDataWidth = m_numPatchesX;
        m_activationDataHeight = m_numPatchesY;
        m_activationDataBufferSize = m_activationDataWidth * m_activationDataHeight * m_activationDataNumChannels;

        allocateBuffers();
    }

    private void allocateBuffers()
    {
        m_filtersBuffer = new float[m_filtersBufferSize];
        Random rand = new Random();
        for (int i = 0; i < m_filtersBufferSize; ++i)
        {
            m_filtersBuffer[i] = 0.01f * (float)rand.nextGaussian();
        }

        m_biasesBuffer = new float[m_biasesBufferSize];
        for (int i = 0; i < m_biasesBufferSize; ++i)
        {
            m_biasesBuffer[i] = 1.0f;
        }

        m_activationDataBuffer = new float[m_activationDataBufferSize];
    }

    /**
     * Gets filters buffer structured for RS neural net implementation.
     * @return Filters buffer structured for RS neural net implementation.
     */
    public float[] getFiltersBufferRS()
    {
        float[] revFiltersBuffer = new float[m_numFilters * m_filterSize * m_inputDataNumChannels];
        for (int filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
        {
            int filterOffset = filterIndex * m_filterSize * m_inputDataNumChannels;
            for (int ch = 0; ch < m_inputDataNumChannels; ++ch)
            {
                for (int px = 0; px < m_filterSize; ++px)
                {
                    revFiltersBuffer[filterOffset + px * m_inputDataNumChannels + ch] = m_filtersBuffer[(ch * m_filterSize + px) * m_numFilters + filterIndex];
                }
            }
        }

        return revFiltersBuffer;
    }

    public float[] getBiasesBuffer()
    {
        return m_biasesBuffer;
    }

    @Override
    public void doForwardProp()
    {
        int inputDataSize = m_inputDataWidth * m_inputDataHeight;
        for (int filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
        {
            final int activationChannelOffset = m_activationDataWidth * m_activationDataHeight * filterIndex;
            for (int channel = 0; channel < m_inputDataNumChannels; ++channel)
            {
                final int filtersChannelOffset = channel * m_numFilters * m_filterSize;
                final int dataChannelOffset = channel * inputDataSize;
                int startY = -m_paddingY;
                for (int patchY = 0; patchY < m_numPatchesY; ++patchY)
                {
                    int startX = -m_paddingX;
                    for (int patchX = 0; patchX < m_numPatchesX; ++patchX)
                    {
                        final int activationDataIndex = activationChannelOffset + patchY * m_numPatchesX + patchX;
                        if (channel == 0)
                        {
                            m_activationDataBuffer[activationDataIndex] = 0.0f;
                        }
                        for (int currY = startY; currY < startY + m_filterHeight; ++currY)
                        {
                            for (int currX = startX; currX < startX + m_filterWidth; ++currX)
                            {
                                if (currY >= 0 && currY < m_inputDataHeight && currX >= 0 && currX < m_inputDataWidth)
                                {
                                    m_activationDataBuffer[activationDataIndex] +=
                                            m_filtersBuffer[filtersChannelOffset + ((currY - startY) * m_filterWidth + currX - startX) * m_numFilters + filterIndex] *
                                            m_inputDataBuffer[dataChannelOffset + currY * m_inputDataWidth + currX];
                                }
                            }
                        }
                        startX += m_stride;
                    }
                    startY += m_stride;
                }
            }
        }

        final int width = m_numPatchesY * m_numPatchesX;
        for (int filterIndex = 0; filterIndex < m_numFilters; ++filterIndex)
        {
            for (int i = 0; i < width; ++i)
            {
                m_activationDataBuffer[filterIndex * width + i] += m_biasesBuffer[filterIndex];
            }
        }
    }
}
