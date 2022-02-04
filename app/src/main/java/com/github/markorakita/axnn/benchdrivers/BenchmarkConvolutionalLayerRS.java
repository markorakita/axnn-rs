package com.github.markorakita.axnn.benchdrivers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.ActivationFunction;
import com.github.markorakita.axnn_rs.neuralnet.layers.ConvolutionalLayerRS;

/**
 * Benchmarks convolutional layer with RS implementation.
 */
public class BenchmarkConvolutionalLayerRS extends BenchmarkDriverRS
{
	/**
	 * Layers parameters.
	 */
	private static final int c_inputNumChannels = 4;
	private static final int c_inputDataWidth = 224;
	private static final int c_inputDataHeight = 224;
	private static final int c_numFilters = 64;
	private static final int c_filterWidth = 11;
	private static final int c_filterHeight = 11;
	private static final int c_paddingX = 1;
	private static final int c_paddingY = 1;
	private static final int c_stride = 4;

	/**
	 * Network convolutional layer with RS implementation.
	 */
	private final ConvolutionalLayerRS m_convLayerRS;

	/**
	 * Constructor.
	 */
	public BenchmarkConvolutionalLayerRS(RenderScript rsContext)
	{
		super(rsContext);

		// Creating layer.
		m_convLayerRS = new ConvolutionalLayerRS(m_rsContext, c_inputNumChannels, c_inputDataWidth, c_inputDataHeight, c_numFilters,
				c_filterWidth, c_filterHeight, c_paddingX, c_paddingY, c_stride, ActivationFunction.ActivationFunctionType.Linear);
		m_convLayerRS.loadFilters(generateRandomBuffer(m_convLayerRS.getFiltersBufferSize()));
		m_convLayerRS.loadBiases(generateRandomBuffer(m_convLayerRS.getBiasesBufferSize()));

		// Allocating input data buffer.
		Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
		int inputDataBufferSize = c_inputNumChannels * c_inputDataWidth * c_inputDataHeight;
		tb.setX(inputDataBufferSize / 4);
		Allocation inputDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		// Generate inputs.
		inputDataBuffer.copyFrom(generateRandomBuffer(inputDataBufferSize));
		m_convLayerRS.setInputDataBuffer(inputDataBuffer);
	}

	/**
	 * Benchmarks convolutional layer with RS implementation.
	 * @return Benchmark results.
	 */
	@WorkerThread
	@Override
	public String executeBenchmark()
	{
		float executionTimeAvg = calculateAverageFpropTime(m_convLayerRS);

		return "<font color=#00FF00>Conv layer fprop took in average: " + executionTimeAvg + "ms.</font><br>";
	}
}
