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
 * From Wikipedia:
 *
 * The Convolutional layer is the core building block of a CNN. The layer's parameters consist of a set of learnable filters (or kernels),
 * which have a small receptive field, but extend through the full depth of the input volume. During the forward pass, each filter is convolved across
 * the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional
 * activation map of that filter. As a result, the network learns filters that activate when they see some specific type of feature at some spatial
 * position in the input.
 */
public class ConvolutionalLayerRS extends LayerRS
{
	/**
	 * Convolutional layer RS kernel.
	 */
	private final ScriptC_convolutionallayer m_kernel;

	/**
	 * Filters buffer, in channel-major order.
	 */
	private Allocation m_filtersBuffer;

	/**
	 * Filters buffer size.
	 */
	private final int m_filtersBufferSize;

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
	 * @param numFilters Number of convolutional filters.
	 * @param filterWidth Filters width.
	 * @param filterHeight Filters height.
	 * @param paddingX Padding in dimension X.
	 * @param paddingY Padding in dimension Y.
	 * @param stride Stride for patching.
	 * @param activationFunctionType Activation function to use.
	 */
	public ConvolutionalLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight, int numFilters, int filterWidth,
								int filterHeight, int paddingX, int paddingY, int stride,
								ActivationFunction.ActivationFunctionType activationFunctionType)
	{
		this(rsContext, inputNumChannels, inputDataWidth, inputDataHeight, numFilters, filterWidth, filterHeight, paddingX, paddingY, stride,
				activationFunctionType, 0.f);
	}

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 * @param inputNumChannels Input data number of channels.
	 * @param inputDataWidth Width of input data.
	 * @param inputDataHeight Height of input data.
	 * @param numFilters Number of convolutional filters.
	 * @param filterWidth Filters width.
	 * @param filterHeight Filters height.
	 * @param paddingX Padding in dimension X.
	 * @param paddingY Padding in dimension Y.
	 * @param stride Stride for patching.
	 * @param activationFunctionType Activation function to use.
	 * @param activationAlpha Alpha parameter of activation function, if it uses one.
	 */
	public ConvolutionalLayerRS(@NonNull RenderScript rsContext, int inputNumChannels, int inputDataWidth, int inputDataHeight, int numFilters, int filterWidth,
								int filterHeight, int paddingX, int paddingY, int stride,
								ActivationFunction.ActivationFunctionType activationFunctionType, float activationAlpha)
	{
		m_kernel = new ScriptC_convolutionallayer(rsContext);

		m_inputDataNumChannels = inputNumChannels;
		m_kernel.set_inputNumChannels(inputNumChannels);
		m_inputDataWidth = inputDataWidth;
		m_kernel.set_inputDataWidth(inputDataWidth);
		m_inputDataHeight = inputDataHeight;
		m_kernel.set_inputDataHeight(inputDataHeight);
		m_inputDataBufferSize = m_inputDataNumChannels * m_inputDataWidth * m_inputDataHeight;

		m_kernel.set_numFilters(numFilters);
		m_kernel.set_filterWidth(filterWidth);
		m_kernel.set_filterHeight(filterHeight);
		m_kernel.set_paddingX(paddingX);
		m_kernel.set_paddingY(paddingY);
		m_kernel.set_stride(stride);
		int numPatchesX = 1 + (int)Math.ceil((double)(2 * paddingX + m_inputDataWidth - filterWidth) / stride);
		m_kernel.set_numPatchesX(numPatchesX);
		int numPatchesY = 1 + (int)Math.ceil((double)(2 * paddingY + m_inputDataHeight - filterHeight) / stride);
		m_kernel.set_numPatchesY(numPatchesY);

		m_activationNumChannels = numFilters;
		m_activationDataWidth = numPatchesX;
		m_activationDataHeight = numPatchesY;
		m_activationDataBufferSize = numFilters * m_activationDataWidth * m_activationDataHeight;

		m_filtersBufferSize = numFilters * filterWidth * filterHeight * m_inputDataNumChannels;
		m_biasesBufferSize = numFilters;

		m_activationFunction = new ActivationFunction(rsContext, activationFunctionType, activationAlpha);

		allocateBuffers(rsContext);
	}

	/**
	 * Allocates local buffers.
	 * @param rsContext Renderscript context.
	 */
	private void allocateBuffers(RenderScript rsContext)
	{
		Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
		tb.setX(m_filtersBufferSize / 4);
		m_filtersBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		tb.setX(m_biasesBufferSize / 4);
		m_biasesBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		tb.setX(m_activationDataBufferSize / 4);
		m_preactivationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
		m_activationDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);
	}

	/**
	 * Gets size of the filters buffer.
	 */
	public int getFiltersBufferSize()
	{
		return m_filtersBufferSize;
	}

	/**
	 * Gets size of the biases buffer.
	 */
	public int getBiasesBufferSize()
	{
		return m_biasesBufferSize;
	}

	/**
	 * Loads filters from model.
	 * @param modelStream Model input stream.
	 * @param bigEndian Should we load filters in big endian or small endian.
	 */
	@WorkerThread
	public void loadFilters(@NonNull InputStream modelStream, boolean bigEndian) throws IOException
	{
		int bufferSizeToLoad = m_filtersBufferSize;

		if (m_inputDataNumChannels == 4)
		{
			// In this case we need to load 1/4 smaller buffer, since we will pad 4th channel with zeros.
			// TODO: revise when we support 4 channel images like ARGB.
			bufferSizeToLoad *= 0.75f;
		}

		float[] filtersBuffer = IOUtils.readFloatBufferFromStream(bufferSizeToLoad, modelStream, bigEndian);
		loadFilters(filtersBuffer);
	}

	/**
	 * Loads filters from host buffer.
	 * @param filtersBuffer Host filters buffer.
	 */
	@WorkerThread
	public void loadFilters(@NonNull float[] filtersBuffer)
	{
		// In this case we need to pad filters buffer with zeros.
		// TODO: revise when we support 4 channel images like ARGB.
		if (m_inputDataNumChannels == 4 && filtersBuffer.length < m_filtersBufferSize)
		{
			float[] paddedFiltersBuffer = new float[m_filtersBufferSize];
			for (int i = 0, j = 0; i < m_filtersBufferSize; ++i)
			{
				paddedFiltersBuffer[i] = i % 4 == 3 ? 0.f : filtersBuffer[j++];
			}

			m_filtersBuffer.copyFrom(paddedFiltersBuffer);
		}
		else
		{
			m_filtersBuffer.copyFrom(filtersBuffer);
		}

		m_kernel.set_filtersBuffer(m_filtersBuffer);
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
		m_kernel.forEach_propagate(m_preactivationDataBuffer);
	}

	@WorkerThread
	private void calculateActivations()
	{
		m_activationDataBuffer = m_activationFunction.applyActivation(m_preactivationDataBuffer, m_activationDataBuffer, m_activationDataBufferSize);
	}
}
