package com.github.markorakita.axnn_rs.neuralnet;

import android.content.Context;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Random;

import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ActivationFunctionTest
{
    private static final int c_bufferSizeBy1 = 113;
    private static final int c_bufferSizeBy4 = 128;
    private static final float c_activationAlpha = 0.01f;

    private RenderScript m_rsContext;

    private float[] m_preactivationDataBF;
    private float[] m_activationDataBF;

    private Allocation m_preactivationDataBuffer;
    private Allocation m_activationDataBuffer;

    @Before
    public void setupRS()
    {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        m_rsContext = RenderScript.create(appContext);
    }

    @Test
    public void testApplyReLUActivation_By1()
    {
        testApplyActivation(c_bufferSizeBy1, ActivationFunction.ActivationFunctionType.ReLU, 0.000001f);
    }

    @Test
    public void testApplyReLUActivation_By4()
    {
        testApplyActivation(c_bufferSizeBy4, ActivationFunction.ActivationFunctionType.ReLU, 0.000001f);
    }

    @Test
    public void testApplyELUActivation_By1()
    {
        testApplyActivation(c_bufferSizeBy1, ActivationFunction.ActivationFunctionType.ELU, 0.00001f);
    }

    @Test
    public void testApplyELUActivation_By4()
    {
        testApplyActivation(c_bufferSizeBy4, ActivationFunction.ActivationFunctionType.ELU, 0.00001f);
    }

    @Test
    public void testApplyLeakyReLUActivation_By1()
    {
        testApplyActivation(c_bufferSizeBy1, ActivationFunction.ActivationFunctionType.LeakyReLU, 0.000001f);
    }

    @Test
    public void testApplyLeakyReLUActivation_By4()
    {
        testApplyActivation(c_bufferSizeBy4, ActivationFunction.ActivationFunctionType.LeakyReLU, 0.000001f);
    }

    @Test
    public void testApplySigmoidActivation()
    {
        testApplyActivation(c_bufferSizeBy1, ActivationFunction.ActivationFunctionType.Sigmoid, 0.0005f);
    }

    @Test
    public void testApplySigmoidActivation_By4()
    {
        testApplyActivation(c_bufferSizeBy4, ActivationFunction.ActivationFunctionType.Sigmoid, 0.0005f);
    }

    @Test
    public void testApplyTanhActivation()
    {
        testApplyActivation(c_bufferSizeBy1, ActivationFunction.ActivationFunctionType.Tanh, 0.0008f);
    }

    @Test
    public void testApplyTanhActivation_By4()
    {
        testApplyActivation(c_bufferSizeBy4, ActivationFunction.ActivationFunctionType.Tanh, 0.0006f);
    }

    private void testApplyActivation(int bufferSize, ActivationFunction.ActivationFunctionType activationFunctionType, float comparisonDelta)
    {
        allocateBuffers(bufferSize);

        ActivationFunction activationFunction = new ActivationFunction(m_rsContext, activationFunctionType, c_activationAlpha);
        applyActivation(bufferSize, activationFunction, activationFunctionType);

        float[] activationData = new float[bufferSize];
        m_activationDataBuffer.copyTo(activationData);

        assertArrayEquals(m_activationDataBF, activationData, comparisonDelta);
    }

    private void allocateBuffers(int bufferSize)
    {
        Type.Builder tb = new Type.Builder(m_rsContext, Element.F32(m_rsContext));
        tb.setX(bufferSize);
        m_preactivationDataBuffer = Allocation.createTyped(m_rsContext, tb.create(), Allocation.USAGE_SCRIPT);

        if (bufferSize % 4 == 0)
        {
            Type.Builder tb4 = new Type.Builder(m_rsContext, Element.F32_4(m_rsContext));
            tb4.setX(bufferSize / 4);
            m_activationDataBuffer = Allocation.createTyped(m_rsContext, tb4.create(), Allocation.USAGE_SCRIPT);
        }
        else
        {
            m_activationDataBuffer = Allocation.createTyped(m_rsContext, tb.create(), Allocation.USAGE_SCRIPT);
        }

        m_preactivationDataBF = new float[bufferSize];
        m_activationDataBF = new float[bufferSize];
        Random rand = new Random();
        for (int i = 0; i < bufferSize; ++i)
        {
            m_preactivationDataBF[i] = 3.f * (float)rand.nextGaussian();
        }

        m_preactivationDataBuffer.copyFrom(m_preactivationDataBF);
    }

    private void applyActivation(int bufferSize, ActivationFunction activationFunction, ActivationFunction.ActivationFunctionType activationFunctionType)
    {
        m_activationDataBuffer = activationFunction.applyActivation(m_preactivationDataBuffer, m_activationDataBuffer, bufferSize);

        for (int i = 0; i < bufferSize; ++i)
        {
            if (activationFunctionType == ActivationFunction.ActivationFunctionType.ReLU)
            {
                m_activationDataBF[i] = Math.max(m_preactivationDataBF[i], 0.f);
            }
            else if (activationFunctionType == ActivationFunction.ActivationFunctionType.ELU)
            {
                m_activationDataBF[i] = m_preactivationDataBF[i] >= 0.f ? m_preactivationDataBF[i] :
                        c_activationAlpha * ((float)Math.exp(m_preactivationDataBF[i]) - 1.f);
            }
            else if (activationFunctionType == ActivationFunction.ActivationFunctionType.LeakyReLU)
            {
                m_activationDataBF[i] = m_preactivationDataBF[i] >= 0.f ? m_preactivationDataBF[i] :
                        c_activationAlpha * m_preactivationDataBF[i];
            }
            else if (activationFunctionType == ActivationFunction.ActivationFunctionType.Sigmoid)
            {
                m_activationDataBF[i] = m_preactivationDataBF[i] >= 0.f ?
                        1.f / (1.f + (float)Math.exp(-m_preactivationDataBF[i])) :
                        (1.f - 1.f / (1.f + (float)Math.exp(m_preactivationDataBF[i])));
            }
            else if (activationFunctionType == ActivationFunction.ActivationFunctionType.Tanh)
            {
                m_activationDataBF[i] = m_preactivationDataBF[i] >= 0.f ?
                        (2.f / (1.f + (float)Math.exp(-2.f * m_preactivationDataBF[i])) - 1.f) :
                        (1.f - 2.f / (1.f + (float)Math.exp(2.f * m_preactivationDataBF[i])));
            }
        }
    }
}
