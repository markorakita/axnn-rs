package com.github.markorakita.axnn.benchdrivers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.layers.DropoutLayerRS;

/**
 * Benchmarks dropout layer with RS implementation.
 */
public class BenchmarkDropoutLayerRS extends BenchmarkDriverRS
{
    /**
     * Layers parameters.
     */
    private static final int c_inputNumChannels = 1;
    private static final int c_inputDataWidth = 1024;
    private static final int c_inputDataHeight = 1;
    private static final float c_dropProbability = 0.5f;

    /**
     * Network dropout layer with RS implementation.
     */
    private final DropoutLayerRS m_dropoutLayerRS;

    /**
     * Constructor.
     */
    public BenchmarkDropoutLayerRS(RenderScript rsContext)
    {
        super(rsContext);

        // Creating layer.
        m_dropoutLayerRS = new DropoutLayerRS(m_rsContext, c_inputNumChannels, c_inputDataWidth, c_inputDataHeight, c_dropProbability);

        // Allocating input data buffer.
        Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
        int inputDataBufferSize = c_inputNumChannels * c_inputDataWidth * c_inputDataHeight;
        tb.setX(inputDataBufferSize / 4);
        Allocation inputDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

        // Generate inputs.
        inputDataBuffer.copyFrom(generateRandomBuffer(inputDataBufferSize));
        m_dropoutLayerRS.setInputDataBuffer(inputDataBuffer);
    }

    /**
     * Benchmarks dropout layer with RS implementation.
     * @return Benchmark results.
     */
    @WorkerThread
    @Override
    public String executeBenchmark()
    {
        float executionTimeAvg = calculateAverageFpropTime(m_dropoutLayerRS);

        return "<font color=#00FF00>Dropout layer fprop took in average: " + executionTimeAvg + "ms.</font><br>";
    }
}
