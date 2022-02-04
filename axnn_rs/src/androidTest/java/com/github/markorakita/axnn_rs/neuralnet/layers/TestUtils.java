package com.github.markorakita.axnn_rs.neuralnet.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

import android.renderscript.Allocation;

class TestUtils
{
    public static void checkCorrectness(float[] cpuLayerActivationBuffer, Allocation rsLayerActivationBuffer, float threshold)
    {
        float[] rsLayerActivations = new float[cpuLayerActivationBuffer.length];
        rsLayerActivationBuffer.copyTo(rsLayerActivations);

        assertArrayEquals(cpuLayerActivationBuffer, rsLayerActivations, threshold);

        boolean foundValueDifferentFromZero = false;
        for (float value : cpuLayerActivationBuffer)
        {
            if (value != 0.f)
            {
                foundValueDifferentFromZero = true;
                break;
            }
        }

        assertTrue(foundValueDifferentFromZero);
    }
}
