package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

/**
 * Max pool layer partitions the input image into a set of regions, and for each such region outputs the maximum value of input activity.
 * It helps to reduce dimensionality, but also learns model to be more invariant to translation.
 */
public class MaxPoolLayerRS extends LayerRS
{
	/**
	 * MaxPool layer RS kernel.
	 */
	private final ScriptC_maxpoollayer m_kernel;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param inputNumChannels Input data number of channels.
	 * @param inputDataWidth Width of input data.
	 * @param inputDataHeight Height of input data.
	 * @param unitWidth Width of the pooling unit.
	 * @param unitHeight Height of the pooling unit.
	 * @param paddingX Padding in dimension X.
	 * @param paddingY Padding in dimension Y.
	 * @param unitStride Stride of the pooling unit.
	 */
	public MaxPoolLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight, int unitWidth, int unitHeight,
						  int paddingX, int paddingY, int unitStride)
	{
		m_kernel = new ScriptC_maxpoollayer(rsContext);

		m_inputDataNumChannels = inputNumChannels;
		m_kernel.set_inputNumChannels(m_inputDataNumChannels);
		m_inputDataWidth = inputDataWidth;
		m_kernel.set_inputDataWidth(inputDataWidth);
		m_inputDataHeight = inputDataHeight;
		m_kernel.set_inputDataHeight(inputDataHeight);
		m_inputDataBufferSize = m_inputDataNumChannels * m_inputDataWidth * m_inputDataHeight;

		m_kernel.set_unitWidth(unitWidth);
		m_kernel.set_unitHeight(unitHeight);
		m_kernel.set_paddingX(paddingX);
		m_kernel.set_paddingY(paddingY);
		m_kernel.set_unitStride(unitStride);

		m_activationNumChannels = inputNumChannels;
		m_activationDataWidth = 1 + (int)Math.ceil((double)(paddingX + m_inputDataWidth - unitWidth) / unitStride);
		m_kernel.set_numUnitsX(m_activationDataWidth);
		m_activationDataHeight = 1 + (int)Math.ceil((double)(paddingY + m_inputDataHeight - unitHeight) / unitStride);
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
