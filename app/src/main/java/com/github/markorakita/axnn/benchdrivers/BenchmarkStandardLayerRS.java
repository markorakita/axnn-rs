package com.github.markorakita.axnn.benchdrivers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;

import androidx.annotation.WorkerThread;

import com.github.markorakita.axnn_rs.neuralnet.ActivationFunction;
import com.github.markorakita.axnn_rs.neuralnet.layers.StandardLayerRS;

/**
 * Benchmarks standard layer with RS implementation.
 */
public class BenchmarkStandardLayerRS extends BenchmarkDriverRS
{
	/**
	 * Layers parameters.
	 */
	private static final int c_inputNumChannels = 256;
	private static final int c_inputDataWidth = 6;
	private static final int c_inputDataHeight = 6;
	private static final int c_numNeurons = 4096;

	/**
	 * Network standard layer with RS implementation.
	 */
	private final StandardLayerRS m_standardLayerRS;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 */
	public BenchmarkStandardLayerRS(RenderScript rsContext)
	{
		super(rsContext);

		// Creating layer.
		m_standardLayerRS = new StandardLayerRS(m_rsContext, c_inputNumChannels, c_inputDataWidth, c_inputDataHeight, c_numNeurons,
				ActivationFunction.ActivationFunctionType.Linear);
		m_standardLayerRS.loadWeights(generateRandomBuffer(m_standardLayerRS.getWeightsBufferSize()));
		m_standardLayerRS.loadBiases(generateRandomBuffer(m_standardLayerRS.getBiasesBufferSize()));

		// Allocating input data buffer.
		Type.Builder tb = new Type.Builder(rsContext, Element.F32_4(rsContext));
		int inputDataBufferSize = c_inputNumChannels * c_inputDataWidth * c_inputDataHeight;
		tb.setX(inputDataBufferSize / 4);
		Allocation inputDataBuffer = Allocation.createTyped(rsContext, tb.create(), Allocation.USAGE_SCRIPT);

		// Generate inputs.
		inputDataBuffer.copyFrom(generateRandomBuffer(inputDataBufferSize));
		m_standardLayerRS.setInputDataBuffer(inputDataBuffer);
	}

	/**
	 * Benchmarks standard layer with RS implementation.
	 * @return Benchmark results.
	 */
	@WorkerThread
	@Override
	public String executeBenchmark()
	{
		float executionTimeAvg = calculateAverageFpropTime(m_standardLayerRS);

		return "<font color=#00FF00>Standard layer fprop took in average: " + executionTimeAvg + "ms.</font><br>";
	}
}
