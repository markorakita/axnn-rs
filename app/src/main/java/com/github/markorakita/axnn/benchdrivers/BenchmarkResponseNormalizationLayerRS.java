package com.github.markorakita.axnn.benchdrivers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.layers.ResponseNormalizationLayerRS;

/**
 * Benchmarks response normalization layer with RS implementation.
 */
public class BenchmarkResponseNormalizationLayerRS extends BenchmarkDriverRS
{
	/**
	 * Layers parameters.
	 */
	private static final int c_inputNumChannels = 4;
	private static final int c_inputDataWidth = 224;
	private static final int c_inputDataHeight = 224;
	private static final int c_depth = 5;
	private static final float c_bias = 2;
	private static final float c_alphaCoeff = 0.0001f;
	private static final float c_betaCoeff = 0.75f;

	/**
	 * Network response normalization layer with RS implementation.
	 */
	private final ResponseNormalizationLayerRS m_reNormLayerRS;

	/**
	 * Constructor.
	 */
	public BenchmarkResponseNormalizationLayerRS(RenderScript rsContext)
	{
		super(rsContext);

		// Creating layer.
		m_reNormLayerRS = new ResponseNormalizationLayerRS(m_rsContext, c_inputNumChannels, c_inputDataWidth, c_inputDataHeight, c_depth, c_bias,
				c_alphaCoeff, c_betaCoeff);

		// Allocating input data buffer.
		Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
		int inputDataBufferSize = c_inputNumChannels * c_inputDataWidth * c_inputDataHeight;
		tb.setX(inputDataBufferSize / 4);
		Allocation inputDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		// Generate inputs.
		inputDataBuffer.copyFrom(generateRandomBuffer(inputDataBufferSize));
		m_reNormLayerRS.setInputDataBuffer(inputDataBuffer);
	}

	/**
	 * Benchmarks response normalization layer with RS implementation.
	 * @return Benchmark results.
	 */
	@WorkerThread
	@Override
	public String executeBenchmark()
	{
		float executionTimeAvg = calculateAverageFpropTime(m_reNormLayerRS);

		return "<font color=#00FF00>ReNorm layer fprop took in average: " + executionTimeAvg + "ms.</font><br>";
	}
}
