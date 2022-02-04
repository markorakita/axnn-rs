package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.content.Context;
import android.renderscript.RenderScript;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.DropoutLayerCPU;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.InputLayerCPU;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class DropoutLayerTest
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
        final float dropProbability = 0.7f;

        InputLayerCPU inputLayerCPU = new InputLayerCPU(m_rsContext, inputDataWidth, inputDataHeight, inputNumChannels);
        inputLayerCPU.generateInputsFromNormalDistribution(0.f, 0.5f);
        inputLayerCPU.doForwardProp();

        DropoutLayerCPU dropoutLayerCPU = new DropoutLayerCPU(inputNumChannels, inputDataWidth, inputDataHeight, dropProbability);
        dropoutLayerCPU.setInputDataBuffer(inputLayerCPU.getActivationDataBuffer());
        dropoutLayerCPU.doForwardProp();

        DropoutLayerRS dropoutLayer = new DropoutLayerRS(m_rsContext, inputNumChannels, inputDataWidth, inputDataHeight, dropProbability);
        dropoutLayer.setInputDataBuffer(inputLayerCPU.getActivationDataBufferRS());
        dropoutLayer.doForwardProp();
        m_rsContext.finish();

        TestUtils.checkCorrectness(dropoutLayerCPU.getActivationDataBuffer(), dropoutLayer.getActivationDataBuffer(), 0.0001f);
    }
}
