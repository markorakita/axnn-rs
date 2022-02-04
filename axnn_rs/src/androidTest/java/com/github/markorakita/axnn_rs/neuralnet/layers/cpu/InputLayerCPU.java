package com.github.markorakita.axnn_rs.neuralnet.layers.cpu;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import java.util.Random;

public class InputLayerCPU extends LayerCPU
{
    private Allocation m_activationDataBufferRS;

    public InputLayerCPU(RenderScript rsContext, int inputDataWidth, int inputDataHeight, int inputDataNumChannels)
    {
        m_inputDataWidth = m_activationDataWidth = inputDataWidth;
        m_inputDataHeight = m_activationDataHeight = inputDataHeight;
        m_inputDataNumChannels = m_activationDataNumChannels = inputDataNumChannels;
        m_inputDataBufferSize = m_activationDataBufferSize = inputDataWidth * inputDataHeight * inputDataNumChannels;

        allocateBuffers(rsContext);
    }

    private void allocateBuffers(RenderScript rsContext)
    {
        m_activationDataBuffer = new float[m_activationDataBufferSize];

        Type.Builder tb;
        if (m_activationDataBufferSize % 4 == 0)
        {
            tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
            tb.setX(m_activationDataBufferSize / 4);
        }
        else
        {
            tb = new Type.Builder(rsContext, Element.F32(rsContext));
            tb.setX(m_activationDataBufferSize);
        }
        m_activationDataBufferRS = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
    }

    public void generateInputsFromUniformDistribution(float rangeStart, float rangeEnd)
    {
        Random rand = new Random();
        for (int i = 0; i < m_activationDataBufferSize; ++i)
        {
            m_activationDataBuffer[i] = rangeStart + (rangeEnd - rangeStart) * rand.nextFloat();
        }
    }

    public void generateInputsFromNormalDistribution(float mean, float stDev)
    {
        Random rand = new Random();
        for (int i = 0; i < m_activationDataBufferSize; ++i)
        {
            m_activationDataBuffer[i] = mean + stDev * (float)rand.nextGaussian();
        }
    }

    @Override
    public void doForwardProp()
    {
        m_activationDataBufferRS.copyFrom(m_activationDataBuffer);
    }

    public Allocation getActivationDataBufferRS()
    {
        return m_activationDataBufferRS;
    }
}
