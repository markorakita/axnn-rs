package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.content.Context;
import android.renderscript.RenderScript;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.github.markorakita.axnn_rs.neuralnet.ActivationFunction;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.InputLayerCPU;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.StandardLayerCPU;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class StandardLayerTest
{
    private RenderScript m_rsContext;

    private StandardLayerCPU m_standardLayerCPU;

    private StandardLayerRS m_standardLayer;

    @Before
    public void setupRS()
    {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        m_rsContext = RenderScript.create(appContext);
    }

    @Test
    public void testForwardPropCorrectness_By1()
    {
        testForwardPropCorrectness(113, 32);
    }

    @Test
    public void testForwardPropCorrectness_By4()
    {
        testForwardPropCorrectness(128, 30);
    }

    private void testForwardPropCorrectness(int inputDataSize, int numNeurons)
    {
        doForwardProp(inputDataSize, numNeurons);

        TestUtils.checkCorrectness(m_standardLayerCPU.getActivationDataBuffer(), m_standardLayer.getActivationDataBuffer(), 0.0001f);
    }

    private void doForwardProp(int inputDataSize, int numNeurons)
    {
        InputLayerCPU inputLayerCPU = new InputLayerCPU(m_rsContext, inputDataSize, 1, 1);
        inputLayerCPU.generateInputsFromUniformDistribution(-128.0f, 127.0f);
        inputLayerCPU.doForwardProp();

        m_standardLayerCPU = new StandardLayerCPU(inputLayerCPU.getActivationDataNumChannels(), inputLayerCPU.getActivationDataWidth(),
                inputLayerCPU.getActivationDataHeight(), numNeurons);
        m_standardLayerCPU.setInputDataBuffer(inputLayerCPU.getActivationDataBuffer());
        m_standardLayerCPU.doForwardProp();

        m_standardLayer = new StandardLayerRS(m_rsContext, inputLayerCPU.getActivationDataNumChannels(), inputLayerCPU.getActivationDataWidth(),
                inputLayerCPU.getActivationDataHeight(), numNeurons, ActivationFunction.ActivationFunctionType.Linear);
        m_standardLayer.loadWeights(m_standardLayerCPU.getWeightsBuffer());
        m_standardLayer.loadBiases(m_standardLayerCPU.getBiasesBuffer());
        m_standardLayer.setInputDataBuffer(inputLayerCPU.getActivationDataBufferRS());
        m_standardLayer.doForwardProp();
        m_rsContext.finish();
    }
}
