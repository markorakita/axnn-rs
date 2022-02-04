package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

/**
 * Response normalization layer implements a form of lateral inhibition inspired by the type found in real neurons,
 * creating competition for big activities amongst neuron outputs computed using different kernels.
 *
 * It is implemented by formula:
 *      activation[i] = input[i] / (bias + (alpha / depth) * sum[j](input[j] ^ 2)) ^ beta
 */
public class ResponseNormalizationLayerRS extends LayerRS
{
	/**
	 * Response normalization layer RS kernel.
	 */
	private final ScriptC_responsenormalizationlayer m_kernel;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param inputNumChannels Input data number of channels.
	 * @param inputDataWidth Width of input data.
	 * @param inputDataHeight Height of input data.
	 * @param depth Depth of normalization.
	 * @param bias Normalization bias.
	 * @param alphaCoeff Normalization alpha coefficient.
	 * @param betaCoeff Normalization beta coefficient.
	 */
	public ResponseNormalizationLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight,
										int depth, float bias, float alphaCoeff, float betaCoeff)
	{
		m_kernel = new ScriptC_responsenormalizationlayer(rsContext);

		m_inputDataNumChannels = inputNumChannels;
		m_kernel.set_inputNumChannels(m_inputDataNumChannels);
		m_inputDataWidth = inputDataWidth;
		m_kernel.set_inputDataWidth(inputDataWidth);
		m_inputDataHeight = inputDataHeight;
		m_kernel.set_inputDataHeight(inputDataHeight);
		m_inputDataBufferSize = m_inputDataNumChannels * m_inputDataWidth * m_inputDataHeight;

		m_kernel.set_depth(depth);
		m_kernel.set_bias(bias);
		// Adjusting alpha coefficient upfront, according to the formula.
		m_kernel.set_alphaCoeff(alphaCoeff / depth);
		m_kernel.set_betaCoeff(betaCoeff);

		m_activationNumChannels = inputNumChannels;
		m_activationDataWidth = inputDataWidth;
		m_activationDataHeight = inputDataHeight;
		m_activationDataBufferSize = m_inputDataNumChannels * m_activationDataWidth * m_activationDataHeight;

		allocateBuffers(rsContext);
	}

	/**
	 * Allocates local buffers.
	 * @param rsContext Renderscript context.
	 */
	private void allocateBuffers(RenderScript rsContext)
	{
		Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
		tb.setX(m_activationDataBufferSize / 4);
		m_activationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
	}

	/**
	 * Does forward propagation through layer.
	 */
	@WorkerThread
	@Override
	public void doForwardProp()
	{
		m_kernel.set_inputDataBuffer(m_inputDataBuffer);
		m_kernel.forEach_propagate(m_activationDataBuffer);
	}
}
