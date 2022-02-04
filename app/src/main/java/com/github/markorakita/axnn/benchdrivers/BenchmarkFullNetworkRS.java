package com.github.markorakita.axnn.benchdrivers;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.Looper;
import android.renderscript.RenderScript;
import android.widget.Toast;

import com.github.markorakita.axnn_rs.neuralnet.NeuralNetFactory;
import com.github.markorakita.axnn_rs.neuralnet.NeuralNetRS;

import java.io.IOException;
import java.io.InputStream;

/**
 * Benchmarks neural network with RS implementation.
 */
public class BenchmarkFullNetworkRS extends BenchmarkDriverRS
{
	/**
	 * Neural network with RS implementation.
	 */
	private NeuralNetRS m_neuralNet;

	/**
	 * Test image.
	 */
	private Bitmap m_testImage;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 */
	public BenchmarkFullNetworkRS(RenderScript rsContext)
	{
		super(rsContext);
	}

	/**
	 * Benchmarks neural network with RS implementation.
	 * @return Benchmark results.
	 */
	@Override
	public String executeBenchmark()
	{
		loadNeuralNetwork();
		loadTestImage();

		float executionTimeAvg = 0.f;
		int numTestPasses = 100;
		for (int i = 1; i <= numTestPasses; ++i)
		{
			long beginTime = System.nanoTime();
			try
			{
				m_neuralNet.classifyImage(m_testImage);
			}
			catch (Exception exc)
			{
				return "Test failed with exception: " + exc.getMessage();
			}
			long endTime = System.nanoTime();
			float executionTime = (float) ((double) (endTime - beginTime) / 1000000.0);
			executionTimeAvg += executionTime;
		}

		return "<font color=#00FF00>Prediction took in average: " + (executionTimeAvg / numTestPasses) + "ms.</font><br>";
	}

	private void loadNeuralNetwork()
	{
		/*
		 * TODO: For anyone using this SDK.
		 *       Here you should implement code to create neural network based on your desired architecture, and load it's model from assets.
 		 */

		// For example, here is how to create AlexNet neural network.
		m_neuralNet = NeuralNetFactory.createAlexNetDnn(m_rsContext);

		// And here is how to load its model from assets.
		try (InputStream modelInputStream = m_rsContext.getApplicationContext().getAssets().open("model.xnnm"))
		{
			m_neuralNet.loadModel(modelInputStream, true);
		}
		catch (IOException exc)
		{
			Handler mainHandler = new Handler(Looper.getMainLooper());
			mainHandler.post(() ->
					Toast.makeText(m_rsContext.getApplicationContext(), "Problem loading trained model, exception: " + exc.toString(),
							Toast.LENGTH_LONG).show());
		}
	}

	private void loadTestImage()
	{
		/*
		 * TODO: For anyone using this SDK.
		 *       Here you should add code to load test image on which you want to benchmark, either from assets or from disk.
		 */

		// For example, here is how to load image from assets.
		try
		{
			m_testImage = BitmapFactory.decodeStream(m_rsContext.getApplicationContext().getAssets().open("test_image.jpg"));
		}
		catch (IOException exc)
		{
			Handler mainHandler = new Handler(Looper.getMainLooper());
			mainHandler.post(() ->
					Toast.makeText(m_rsContext.getApplicationContext(), "Problem loading test image, exception: " + exc.toString(),
							Toast.LENGTH_LONG).show());
		}
	}
}
