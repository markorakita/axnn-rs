package com.github.markorakita.axnn.benchdrivers;

import android.renderscript.RenderScript;

import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.layers.LayerRS;

import java.util.Random;

public abstract class BenchmarkDriverRS
{
    private static final int c_numTestPasses = 100;

    /**
     * RenderScript context.
     */
    protected final RenderScript m_rsContext;

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     */
    protected BenchmarkDriverRS(RenderScript rsContext)
    {
        m_rsContext = rsContext;
    }

    @WorkerThread
    public abstract String executeBenchmark();

    @WorkerThread
    protected float calculateAverageFpropTime(LayerRS layer)
    {
        float executionTimeAvg = 0.f;

        for (int i = 1; i <= c_numTestPasses; ++i)
        {
            long beginTime = System.nanoTime();
            layer.doForwardProp();
            m_rsContext.finish();
            long endTime = System.nanoTime();
            float executionTime = (float) ((double) (endTime - beginTime) / 1000000.0);
            executionTimeAvg += executionTime;
        }

        return executionTimeAvg / c_numTestPasses;
    }

    protected static float[] generateRandomBuffer(int bufferSize)
    {
        float[] buffer = new float[bufferSize];
        Random rand = new Random();
        for (int i = 0; i < bufferSize; ++i)
        {
            buffer[i] = (float)rand.nextGaussian();
        }

        return buffer;
    }
}
