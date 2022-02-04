package com.github.markorakita.axnn_rs.neuralnet;

import android.renderscript.Allocation;
import android.renderscript.RenderScript;

public class ActivationFunction
{
    public enum ActivationFunctionType
    {
        Linear,
        ReLU,
        ELU,
        LeakyReLU,
        Sigmoid,
        Tanh
    }

    private final ScriptC_activationfunctions m_kernel;

    private final ActivationFunctionType m_activationFunctionType;

    private final float m_activationAlpha;

    public ActivationFunction(RenderScript rsContext, ActivationFunctionType activationFunctionType, float activationAlpha)
    {
        m_kernel = new ScriptC_activationfunctions(rsContext);
        m_activationFunctionType = activationFunctionType;
        m_activationAlpha = activationAlpha;
    }

    public Allocation applyActivation(Allocation preactivationDataBuffer, Allocation activationDataBuffer, int bufferSize)
    {
        if (m_activationFunctionType == ActivationFunctionType.Linear)
        {
            return preactivationDataBuffer;
        }

        if (m_activationFunctionType == ActivationFunctionType.ReLU)
        {
            applyReLUActivation(preactivationDataBuffer, activationDataBuffer, bufferSize);
        }
        else if (m_activationFunctionType == ActivationFunctionType.ELU)
        {
            applyELUActivation(preactivationDataBuffer, activationDataBuffer, bufferSize);
        }
        else if (m_activationFunctionType == ActivationFunctionType.LeakyReLU)
        {
            applyLeakyReLUActivation(preactivationDataBuffer, activationDataBuffer, bufferSize);
        }
        else if (m_activationFunctionType == ActivationFunctionType.Sigmoid)
        {
            applySigmoidActivation(preactivationDataBuffer, activationDataBuffer, bufferSize);
        }
        else if (m_activationFunctionType == ActivationFunctionType.Tanh)
        {
            applyTanhActivation(preactivationDataBuffer, activationDataBuffer, bufferSize);
        }

        return activationDataBuffer;
    }

    private void applyReLUActivation(Allocation preactivationDataBuffer, Allocation activationDataBuffer, int bufferSize)
    {
        m_kernel.set_preactivationDataBuffer(preactivationDataBuffer);

        if (bufferSize % 4 == 0)
        {
            m_kernel.forEach_applyReLUActivationBy4(activationDataBuffer);
        }
        else
        {
            m_kernel.forEach_applyReLUActivation(activationDataBuffer);
        }
    }

    private void applyELUActivation(Allocation preactivationDataBuffer, Allocation activationDataBuffer, int bufferSize)
    {
        m_kernel.set_preactivationDataBuffer(preactivationDataBuffer);
        m_kernel.set_activationAlpha(m_activationAlpha);

        if (bufferSize % 4 == 0)
        {
            m_kernel.forEach_applyELUActivationBy4(activationDataBuffer);
        }
        else
        {
            m_kernel.forEach_applyELUActivation(activationDataBuffer);
        }
    }

    private void applyLeakyReLUActivation(Allocation preactivationDataBuffer, Allocation activationDataBuffer, int bufferSize)
    {
        m_kernel.set_preactivationDataBuffer(preactivationDataBuffer);
        m_kernel.set_activationAlpha(m_activationAlpha);

        if (bufferSize % 4 == 0)
        {
            m_kernel.forEach_applyLeakyReLUActivationBy4(activationDataBuffer);
        }
        else
        {
            m_kernel.forEach_applyLeakyReLUActivation(activationDataBuffer);
        }
    }

    private void applySigmoidActivation(Allocation preactivationDataBuffer, Allocation activationDataBuffer, int bufferSize)
    {
        m_kernel.set_preactivationDataBuffer(preactivationDataBuffer);

        if (bufferSize % 4 == 0)
        {
            m_kernel.forEach_applySigmoidActivationBy4(activationDataBuffer);
        }
        else
        {
            m_kernel.forEach_applySigmoidActivation(activationDataBuffer);
        }
    }

    private void applyTanhActivation(Allocation preactivationDataBuffer, Allocation activationDataBuffer, int bufferSize)
    {
        m_kernel.set_preactivationDataBuffer(preactivationDataBuffer);

        if (bufferSize % 4 == 0)
        {
            m_kernel.forEach_applyTanhActivationBy4(activationDataBuffer);
        }
        else
        {
            m_kernel.forEach_applyTanhActivation(activationDataBuffer);
        }
    }
}
