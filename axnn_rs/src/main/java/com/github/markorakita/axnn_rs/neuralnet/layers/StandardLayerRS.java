package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.ActivationFunction;
import com.github.markorakita.axnn_rs.neuralnet.internal.utils.IOUtils;

import java.io.IOException;
import java.io.InputStream;

/**
 * Standard neural network layer, with neurons and weights.
 */
public class StandardLayerRS extends LayerRS
{
	/**
	 * Standard layer RS kernel.
	 */
	private final ScriptC_standardlayer m_kernel;

	/**
	 * Weights buffer, in channel-major order.
	 */
	private Allocation m_weightsBuffer;

	/**
	 * Weights buffer size.
	 */
	private final int m_weightsBufferSize;

	/**
	 * Biases buffer.
	 */
	private Allocation m_biasesBuffer;

	/**
	 * Biases buffer size.
	 */
	private final int m_biasesBufferSize;

	/**
	 * Pre-activation data buffer.
	 */
	private Allocation m_preactivationDataBuffer;

	/**
	 * Activation function to use.
	 */
	private final ActivationFunction m_activationFunction;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param inputNumChannels Input data number of channels.
	 * @param inputDataWidth Width of input data.
	 * @param inputDataHeight Height of input data.
	 * @param numNeurons Number of neurons.
	 * @param activationFunctionType Activation function to use.
	 */
	public StandardLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight, int numNeurons,
						   ActivationFunction.ActivationFunctionType activationFunctionType)
	{
		this(rsContext, inputNumChannels, inputDataWidth, inputDataHeight, numNeurons, activationFunctionType, 0.f);
	}

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param inputNumChannels Input data number of channels.
	 * @param inputDataWidth Width of input data.
	 * @param inputDataHeight Height of input data.
	 * @param numNeurons Number of neurons.
	 * @param activationFunctionType Activation function to use.
	 * @param activationAlpha Alpha parameter of activation function, if it uses one.
	 */
	public StandardLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight, int numNeurons,
						   ActivationFunction.ActivationFunctionType activationFunctionType, float activationAlpha)
	{
		m_kernel = new ScriptC_standardlayer(rsContext);

		m_inputDataNumChannels = inputNumChannels;
		m_inputDataWidth = inputDataWidth;
		m_inputDataHeight = inputDataHeight;
		m_inputDataBufferSize = m_inputDataNumChannels * m_inputDataWidth * m_inputDataHeight;
		m_kernel.set_inputDataSize(m_inputDataBufferSize);

		m_activationNumChannels = 1;
		m_activationDataWidth = numNeurons;
		m_activationDataHeight = 1;
		m_activationDataBufferSize = m_activationNumChannels * m_activationDataWidth * m_activationDataHeight;

		m_weightsBufferSize = numNeurons * m_inputDataBufferSize;
		m_biasesBufferSize = numNeurons;

		m_activationFunction = new ActivationFunction(rsContext, activationFunctionType, activationAlpha);

		allocateBuffers(rsContext);
	}

	/**
	 * Allocates local buffers.
	 * @param rsContext Renderscript context.
	 */
	private void allocateBuffers(RenderScript rsContext)
	{
		Type.Builder tb = new Type.Builder(rsContext, Element.F32(rsContext));
		tb.setX(m_biasesBufferSize);
		m_biasesBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		tb.setX(m_activationDataBufferSize);
		m_preactivationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		if (m_inputDataBufferSize % 4 == 0)
		{
			Type.Builder tb4 = new Type.Builder(rsContext, Element.F32_4(rsContext));
			tb4.setX(m_weightsBufferSize / 4);
			m_weightsBuffer = Allocation.createTyped(rsContext, tb4.create(), Allocation.USAGE_SCRIPT);
		}
		else
		{
			tb.setX(m_weightsBufferSize);
			m_weightsBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
		}

		if (m_activationDataBufferSize % 4 == 0)
		{
			Type.Builder tb4 = new Type.Builder(rsContext, Element.F32_4(rsContext));
			tb4.setX(m_activationDataBufferSize / 4);
			m_activationDataBuffer = Allocation.createTyped(rsContext, tb4.create(), Allocation.USAGE_SCRIPT);
		}
		else
		{
			tb.setX(m_activationDataBufferSize);
			m_activationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
		}
	}

	/**
	 * Gets size of the weights buffer.
	 */
	public int getWeightsBufferSize()
	{
		return m_weightsBufferSize;
	}

	/**
	 * Gets size of the biases buffer.
	 */
	public int getBiasesBufferSize()
	{
		return m_biasesBufferSize;
	}

	/**
	 * Loads weights from model.
	 * @param modelStream Model input stream.
	 * @param bigEndian Should we load weights in big endian or small endian.
	 */
	@WorkerThread
	public void loadWeights(@NonNull InputStream modelStream, boolean bigEndian) throws IOException
	{
		float[] weightsBuffer = IOUtils.readFloatBufferFromStream(m_weightsBufferSize, modelStream, bigEndian);
		loadWeights(weightsBuffer);
	}

	/**
	 * Loads weights from host buffer.
	 * @param weightsBuffer Host weights buffer.
	 */
	@WorkerThread
	public void loadWeights(@NonNull float[] weightsBuffer)
	{
		m_weightsBuffer.copyFrom(weightsBuffer);
		m_kernel.set_weightsBuffer(m_weightsBuffer);
	}

	/**
	 * Loads biases from model.
	 * @param modelStream Model input stream.
	 * @param bigEndian Should we load biases in big endian or small endian.
	 */
	@WorkerThread
	public void loadBiases(@NonNull InputStream modelStream, boolean bigEndian) throws IOException
	{
		float[] biasesBuffer = IOUtils.readFloatBufferFromStream(m_biasesBufferSize, modelStream, bigEndian);
		loadBiases(biasesBuffer);
	}

	/**
	 * Loads biases from host buffer.
	 * @param biasesBuffer Host biases buffer.
	 */
	@WorkerThread
	public void loadBiases(@NonNull float[] biasesBuffer)
	{
		m_biasesBuffer.copyFrom(biasesBuffer);
		m_kernel.set_biasesBuffer(m_biasesBuffer);
	}

	/**
	 * Does forward propagation through layer.
	 */
	@WorkerThread
	@Override
	public void doForwardProp()
	{
		calculatePreactivations();
		calculateActivations();
	}

	@WorkerThread
	private void calculatePreactivations()
	{
		m_kernel.set_inputDataBuffer(m_inputDataBuffer);
		if (m_inputDataBufferSize % 4 == 0)
		{
			m_kernel.forEach_calculatePreactivationsBy4(m_preactivationDataBuffer);
		}
		else
		{
			m_kernel.forEach_calculatePreactivations(m_preactivationDataBuffer);
		}
	}

	@WorkerThread
	private void calculateActivations()
	{
		m_activationDataBuffer = m_activationFunction.applyActivation(m_preactivationDataBuffer, m_activationDataBuffer, m_activationDataBufferSize);
	}
}