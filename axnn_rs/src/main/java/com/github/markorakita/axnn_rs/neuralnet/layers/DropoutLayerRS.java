package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

/**
 * Dropout layer provides efficient way to simulate combining multiple trained models to reduce test error and prevent overfitting.
 * It works by dropping each neuron activity with certain probability, preventing complex coadaptations between neurons.
 */
public class DropoutLayerRS extends LayerRS
{
    /**
     * Dropout layer RS kernel.
     */
    private final ScriptC_dropoutlayer m_kernel;

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     * @param inputNumChannels Input data number of channels.
     * @param inputDataWidth Width of input data.
     * @param inputDataHeight Height of input data.
     * @param dropProbability Probability to drop some neuron activation.
     */
    public DropoutLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight, float dropProbability)
    {
        m_kernel = new ScriptC_dropoutlayer(rsContext);

        m_inputDataNumChannels = m_activationNumChannels = inputNumChannels;
        m_inputDataWidth = m_activationDataWidth = inputDataWidth;
        m_inputDataHeight = m_activationDataHeight = inputDataHeight;
        m_inputDataBufferSize = m_activationDataBufferSize = inputNumChannels * inputDataWidth * inputDataHeight;
        m_kernel.set_dropProbability(dropProbability);

        allocateBuffers(rsContext);
    }

    /**
     * Allocates local buffers.
     * @param rsContext Renderscript context.
     */
    private void allocateBuffers(RenderScript rsContext)
    {
        Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
        tb.setX(m_activationDataBufferSize / 4);
        m_activationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
    }

    /**
     * Does forward propagation through layer.
     */
    @WorkerThread
    @Override
    public void doForwardProp()
    {
        m_kernel.set_inputDataBuffer(m_inputDataBuffer);
        m_kernel.forEach_propagate(m_activationDataBuffer);
    }
}
