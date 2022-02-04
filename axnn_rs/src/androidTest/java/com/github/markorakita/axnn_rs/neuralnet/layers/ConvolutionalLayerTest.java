package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.content.Context;
import android.renderscript.RenderScript;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.github.markorakita.axnn_rs.neuralnet.ActivationFunction;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.ConvolutionalLayerCPU;
import com.github.markorakita.axnn_rs.neuralnet.layers.cpu.InputLayerCPU;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class ConvolutionalLayerTest
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
        final int inputDataWidth = 224;
        final int inputDataHeight = 224;
        final int inputNumChannels = 4;
        final int numFilters = 64;
        final int filterWidth = 11;
        final int filterHeight = 11;
        final int paddingX = 1;
        final int paddingY = 1;
        final int stride = 4;

        InputLayerCPU inputLayerCPU = new InputLayerCPU(m_rsContext, inputDataWidth, inputDataHeight, inputNumChannels);
        inputLayerCPU.generateInputsFromUniformDistribution(-128.0f, 127.0f);
        inputLayerCPU.doForwardProp();

        ConvolutionalLayerCPU convolutionalLayerCPU = new ConvolutionalLayerCPU(inputNumChannels, inputDataWidth, inputDataHeight, numFilters, filterWidth, filterHeight,
                paddingX, paddingY, stride);
        convolutionalLayerCPU.setInputDataFromPerPixelBuffer(inputLayerCPU.getActivationDataBuffer());
        convolutionalLayerCPU.doForwardProp();

        ConvolutionalLayerRS convolutionalLayer = new ConvolutionalLayerRS(m_rsContext, inputNumChannels, inputDataWidth, inputDataHeight, numFilters, filterWidth,
                filterHeight, paddingX, paddingY, stride, ActivationFunction.ActivationFunctionType.Linear);
        convolutionalLayer.loadFilters(convolutionalLayerCPU.getFiltersBufferRS());
        convolutionalLayer.loadBiases(convolutionalLayerCPU.getBiasesBuffer());
        convolutionalLayer.setInputDataBuffer(inputLayerCPU.getActivationDataBufferRS());
        convolutionalLayer.doForwardProp();
        m_rsContext.finish();

        TestUtils.checkCorrectness(convolutionalLayerCPU.getActivationDataPerPixelBuffer(), convolutionalLayer.getActivationDataBuffer(), 0.0001f);
    }
}
