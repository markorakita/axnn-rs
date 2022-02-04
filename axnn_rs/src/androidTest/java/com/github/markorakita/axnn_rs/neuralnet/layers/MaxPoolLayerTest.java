package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.content.Context;
import android.renderscript.RenderScript;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.InputLayerCPU;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.MaxPoolLayerCPU;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class MaxPoolLayerTest
{
    private RenderScript m_rsContext;

    @Before
    public void setupRS()
    {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        m_rsContext = RenderScript.create(appContext);
    }

    @Test
    public void testForwardPropCorrectness()
    {
        final int inputDataWidth = 55;
        final int inputDataHeight = 55;
        final int inputNumChannels = 64;
        final int unitWidth = 3;
        final int unitHeight = 3;
        final int paddingX = 0;
        final int paddingY = 0;
        final int unitStride = 2;

        InputLayerCPU inputLayerCPU = new InputLayerCPU(m_rsContext, inputDataWidth, inputDataHeight, inputNumChannels);
        inputLayerCPU.generateInputsFromNormalDistribution(0.f, 0.5f);
        inputLayerCPU.doForwardProp();

        MaxPoolLayerCPU maxPoolLayerCPU = new MaxPoolLayerCPU(inputNumChannels, inputDataWidth, inputDataHeight, unitWidth, unitHeight,
                paddingX, paddingY, unitStride);
        maxPoolLayerCPU.setInputDataFromPerPixelBuffer(inputLayerCPU.getActivationDataBuffer());
        maxPoolLayerCPU.doForwardProp();

        MaxPoolLayerRS maxPoolLayer = new MaxPoolLayerRS(m_rsContext, inputNumChannels, inputDataWidth, inputDataHeight, unitWidth, unitHeight,
                paddingX, paddingY, unitStride);
        maxPoolLayer.setInputDataBuffer(inputLayerCPU.getActivationDataBufferRS());
        maxPoolLayer.doForwardProp();
        m_rsContext.finish();

        TestUtils.checkCorrectness(maxPoolLayerCPU.getActivationDataPerPixelBuffer(), maxPoolLayer.getActivationDataBuffer(), 0.0001f);
    }
}
