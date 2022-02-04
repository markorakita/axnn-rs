package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;

import androidx.annotation.NonNull;

/**
 * Abstract neural network layer.
 */
public abstract class LayerRS
{
	/**
	 * Number of input data channels.
	 */
	protected int m_inputDataNumChannels;

	/**
	 * Input data width.
	 */
	protected int m_inputDataWidth;

	/**
	 * Input data height.
	 */
	protected int m_inputDataHeight;

	/**
	 * Input data buffer.
	 *
	 * RS layers have per-pixel data structure (R1G1B1R2G2B2).
	 */
	protected Allocation m_inputDataBuffer;

	/**
	 * Input data buffer size.
	 */
	protected int m_inputDataBufferSize;

	/**
	 * Number of activation data channels.
	 */
	protected int m_activationNumChannels;

	/**
	 * Activation data width.
	 */
	protected int m_activationDataWidth;

	/**
	 * Activation data height.
	 */
	protected int m_activationDataHeight;

	/**
	 * Activations buffer.
	 */
	protected Allocation m_activationDataBuffer;

	/**
	 * Activations data buffer size.
	 */
	protected int m_activationDataBufferSize;

	/**
	 * Sets input data buffer.
	 * @param inputDataBuffer Input data buffer to set.
	 */
	public void setInputDataBuffer(@NonNull Allocation inputDataBuffer)
	{
		m_inputDataBuffer = inputDataBuffer;
	}

	/**
	 * Gets number of activation channels.
	 * @return Number of activation channels.
	 */
	public int getActivationNumChannels()
	{
		return m_activationNumChannels;
	}

	/**
	 * Gets activation data width.
	 * @return Activation data width.
	 */
	public int getActivationDataWidth()
	{
		return m_activationDataWidth;
	}

	/**
	 * Gets activation data height.
	 * @return Activation data height.
	 */
	public int getActivationDataHeight()
	{
		return m_activationDataHeight;
	}

	/**
	 * Gets activation data buffer.
	 * @return Activation data buffer.
	 */
	public Allocation getActivationDataBuffer()
	{
		return m_activationDataBuffer;
	}

	/**
	 * Gets activation buffer size.
	 * @return Activation buffer size.
	 */
	public int getActivationDataBufferSize()
	{
		return m_activationDataBufferSize;
	}

	/**
	 * Does forward propagation through layer.
	 */
	public abstract void doForwardProp();
}
