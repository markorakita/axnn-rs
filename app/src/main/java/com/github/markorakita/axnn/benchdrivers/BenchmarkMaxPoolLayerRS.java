package com.github.markorakita.axnn.benchdrivers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.layers.MaxPoolLayerRS;

/**
 * Benchmarks max pool layer with RS implementation.
 */
public class BenchmarkMaxPoolLayerRS extends BenchmarkDriverRS
{
	/**
	 * Layers parameters.
	 */
	private static final int c_inputNumChannels = 4;
	private static final int c_inputDataWidth = 224;
	private static final int c_inputDataHeight = 224;
	private static final int c_unitWidth = 3;
	private static final int c_unitHeight = 3;
	private static final int c_paddingX = 0;
	private static final int c_paddingY = 0;
	private static final int c_unitStride = 2;

	/**
	 * Network max pool layer with RS implementation.
	 */
	private final MaxPoolLayerRS m_maxPoolLayerRS;

	/**
	 * Constructor.
	 */
	public BenchmarkMaxPoolLayerRS(RenderScript rsContext)
	{
		super(rsContext);

		// Creating layer.
		m_maxPoolLayerRS = new MaxPoolLayerRS(m_rsContext, c_inputNumChannels, c_inputDataWidth, c_inputDataHeight, c_unitWidth, c_unitHeight,
				c_paddingX, c_paddingY, c_unitStride);

		// Allocating input data buffer.
		Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
		int inputDataBufferSize = c_inputNumChannels * c_inputDataWidth * c_inputDataHeight;
		tb.setX(inputDataBufferSize / 4);
		Allocation inputDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		// Generate inputs.
		inputDataBuffer.copyFrom(generateRandomBuffer(inputDataBufferSize));
		m_maxPoolLayerRS.setInputDataBuffer(inputDataBuffer);
	}

	/**
	 * Benchmarks max pool layer with RS implementation.
	 * @return Benchmark results.
	 */
	@WorkerThread
	@Override
	public String executeBenchmark()
	{
		float executionTimeAvg = calculateAverageFpropTime(m_maxPoolLayerRS);

		return "<font color=#00FF00>MaxPool layer fprop took in average: " + executionTimeAvg + "ms.</font><br>";
	}
}
