package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.NonNull;

/**
 * Softmax layer, calculates soft-max probabilities of each class.
 */
public class SoftMaxLayerRS extends LayerRS
{
    private float[] m_hostInputDataBuffer;

    private float[] m_hostActivationDataBuffer;

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     * @param inputDataSize Size of the input data buffer.
     */
    public SoftMaxLayerRS(@NonNull RenderScript rsContext, int inputDataSize)
    {
        m_inputDataBufferSize = m_activationDataBufferSize = inputDataSize;

        allocateBuffers(rsContext);
    }

    /**
     * Allocates local buffers.
     * @param rsContext Renderscript context.
     */
    private void allocateBuffers(RenderScript rsContext)
    {
        m_hostInputDataBuffer = new float[m_inputDataBufferSize];

        m_hostActivationDataBuffer = new float[m_activationDataBufferSize];

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
        m_activationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
    }

    /**
     * Does forward propagation through layer.
     */
    @Override
    public void doForwardProp()
    {
        m_inputDataBuffer.copyTo(m_hostInputDataBuffer);

        stabilizeInputs();
        calculateSoftMaximums();

        m_activationDataBuffer.copyFrom(m_hostActivationDataBuffer);
    }

    private void stabilizeInputs()
    {
        float inputActivationsMaximum = m_hostInputDataBuffer[0];
        for (float inputData : m_hostInputDataBuffer)
        {
            inputActivationsMaximum = Math.max(inputActivationsMaximum, inputData);
        }

        for (int i = 0; i < m_inputDataBufferSize; ++i)
        {
            m_hostActivationDataBuffer[i] = m_hostInputDataBuffer[i] - inputActivationsMaximum;
        }
    }

    private void calculateSoftMaximums()
    {
        float exponentialsSum = 0.f;
        for (int i = 0; i < m_activationDataBufferSize; ++i)
        {
            m_hostActivationDataBuffer[i] = (float)Math.exp(m_hostActivationDataBuffer[i]);
            exponentialsSum += m_hostActivationDataBuffer[i];
        }

        for (int i = 0; i < m_activationDataBufferSize; ++i)
        {
            m_hostActivationDataBuffer[i] /= exponentialsSum;
        }
    }
}
