package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.NonNull;

/**
 * Input layer loads data for the network propagation.
 */
public class InputLayerRS extends LayerRS
{
	/**
	 * Input data before normalization.
	 */
	protected float[] m_unnormalizedInputDataBuffer;

	/**
	 * Should input data be normalized.
	 */
	private boolean m_normalizeInputData;

	/**
	 * Mean per channel on which to normalize input data.
	 */
	private float[] m_inputDataMeans;

	/**
	 * Standard deviation per channel on which to normalize input data.
	 */
	private float[] m_inputDataStDevs;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param activationDataWidth Expected width of activation data.
	 * @param activationDataHeight Expected height of activation data.
	 * @param activationNumChannels Expected number of channels in activation data.
	 */
	public InputLayerRS(@NonNull RenderScript rsContext, int activationDataWidth, int activationDataHeight, int activationNumChannels)
	{
		m_inputDataWidth = m_activationDataWidth = activationDataWidth;
		m_inputDataHeight = m_activationDataHeight = activationDataHeight;
		m_inputDataNumChannels = m_activationNumChannels = activationNumChannels;
		m_inputDataBufferSize = m_activationDataBufferSize = activationDataWidth * activationDataHeight * activationNumChannels;

		m_normalizeInputData = false;

		// Allocating activation data buffer.
		Type.Builder tb;
		if (m_activationDataBufferSize % 4 == 0)
		{
			tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
			tb.setX(m_activationDataBufferSize / 4);
		}
		else
		{
			tb = new Type.Builder(rsContext, Element.F32(rsContext));
			tb.setX(m_activationDataBufferSize);
		}
		m_activationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
	}

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param activationDataWidth Expected width of activation data.
	 * @param activationDataHeight Expected height of activation data.
	 * @param activationNumChannels Expected number of channels in activation data.
	 * @param inputDataMeans Mean per channel on which to normalize input data.
	 * @param inputDataStDevs Standard deviation per channel on which to normalize input data.
	 */
	public InputLayerRS(@NonNull RenderScript rsContext, int activationDataWidth, int activationDataHeight, int activationNumChannels,
						float[] inputDataMeans, float[] inputDataStDevs)
	{
		this(rsContext, activationDataWidth, activationDataHeight, activationNumChannels);

		m_normalizeInputData = true;
		m_inputDataMeans = inputDataMeans;
		m_inputDataStDevs = inputDataStDevs;
	}

	@Override
	public void doForwardProp()
	{
		if (m_normalizeInputData)
		{
			normalizeInputs();
		}

		m_activationDataBuffer.copyFrom(m_unnormalizedInputDataBuffer);
	}

	private void normalizeInputs()
	{
		for (int i = 0; i < m_unnormalizedInputDataBuffer.length; ++i)
		{
			m_unnormalizedInputDataBuffer[i] = (m_unnormalizedInputDataBuffer[i] - m_inputDataMeans[i % m_inputDataMeans.length]) /
											   m_inputDataStDevs[i % m_inputDataStDevs.length];
		}
	}
}
