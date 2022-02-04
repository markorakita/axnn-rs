package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.content.Context;
import android.renderscript.RenderScript;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.InputLayerCPU;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.ResponseNormalizationLayerCPU;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class ResponseNormalizationLayerTest
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
        final int inputNumChannels = 64;
        final int inputDataWidth = 55;
        final int inputDataHeight = 55;
        final int depth = 5;
        final float bias = 2;
        final float alphaCoeff = 0.0001f;
        final float betaCoeff = 0.75f;

        InputLayerCPU inputLayerCPU = new InputLayerCPU(m_rsContext, inputDataWidth, inputDataHeight, inputNumChannels);
        inputLayerCPU.generateInputsFromNormalDistribution(0.f, 0.5f);
        inputLayerCPU.doForwardProp();

        ResponseNormalizationLayerCPU responseNormalizationLayerCPU = new ResponseNormalizationLayerCPU(inputNumChannels, inputDataWidth,
                inputDataHeight, depth, bias, alphaCoeff, betaCoeff);
        responseNormalizationLayerCPU.setInputDataFromPerPixelBuffer(inputLayerCPU.getActivationDataBuffer());
        responseNormalizationLayerCPU.doForwardProp();

        ResponseNormalizationLayerRS responseNormalizationLayer = new ResponseNormalizationLayerRS(m_rsContext, inputNumChannels, inputDataWidth,
                inputDataHeight, depth, bias, alphaCoeff, betaCoeff);
        responseNormalizationLayer.setInputDataBuffer(inputLayerCPU.getActivationDataBufferRS());
        responseNormalizationLayer.doForwardProp();
        m_rsContext.finish();

        TestUtils.checkCorrectness(responseNormalizationLayerCPU.getActivationDataPerPixelBuffer(),
                responseNormalizationLayer.getActivationDataBuffer(), 0.0005f);
    }
}
