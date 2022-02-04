package com.github.markorakita.axnn;

import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.renderscript.RenderScript;
import android.text.Html;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import com.github.markorakita.axnn.benchdrivers.BenchmarkConvolutionalLayerRS;
import com.github.markorakita.axnn.benchdrivers.BenchmarkDriverRS;
import com.github.markorakita.axnn.benchdrivers.BenchmarkDropoutLayerRS;
import com.github.markorakita.axnn.benchdrivers.BenchmarkFullNetworkRS;
import com.github.markorakita.axnn.benchdrivers.BenchmarkMaxPoolLayerRS;
import com.github.markorakita.axnn.benchdrivers.BenchmarkResponseNormalizationLayerRS;
import com.github.markorakita.axnn.benchdrivers.BenchmarkStandardLayerRS;
import com.github.markorakita.axnn.testdrivers.TestDriverRS;
import com.github.markorakita.axnn.testdrivers.TestFullNetworkRS;

/**
 * Main app activity.
 */
public class MainActivity extends AppCompatActivity
{
	/**
	 * RenderScript context.
	 */
	private RenderScript m_rsContext;

	/**
	 * Called on creation of main activity.
	 * @param savedInstanceState Saved instance state of main activity.
	 */
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		m_rsContext = RenderScript.create(this);

		// Populate layers dropdown.
		Spinner layerChoiceSpinner = findViewById(R.id.layerChoice);
		ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.layer_choice_items, android.R.layout.simple_spinner_item);
		adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
		layerChoiceSpinner.setAdapter(adapter);

		// Set buttons listeners.
		Button testRsBtn = findViewById(R.id.testRsBtn);
		testRsBtn.setOnClickListener(new TestRsBtnHandler(layerChoiceSpinner));
		Button benchRsBtn = findViewById(R.id.benchRsBtn);
		benchRsBtn.setOnClickListener(new BenchRsBtnHandler(layerChoiceSpinner));
	}

	/**
	 * Handles click on button to test RenderScript implementation of the network.
	 */
	private class TestRsBtnHandler implements View.OnClickListener
	{
		/**
		 * Spinner for choosing layer to test or benchmark.
		 */
		private final Spinner m_layerChoiceSpinner;

		/**
		 * Constructor.
		 * @param layerChoiceSpinner Spinner for choosing layer to test or benchmark.
		 */
		TestRsBtnHandler(Spinner layerChoiceSpinner)
		{
			m_layerChoiceSpinner = layerChoiceSpinner;
		}

		/**
		 * Tests RenderScript implementation of the network, for chosen layer.
		 * @param buttonView Button for testing RenderScript implementation of the network.
		 */
		@Override
		public void onClick(View buttonView)
		{
			String selectedLayerName =  m_layerChoiceSpinner.getSelectedItem().toString();

			setupButtons(false);
			findViewById(R.id.resultText).setVisibility(View.GONE);
			findViewById(R.id.progressBar).setVisibility(View.VISIBLE);

			Thread thread = new Thread(() -> executeTest(selectedLayerName));
			thread.start();
		}

		@WorkerThread
		private void executeTest(String selectedLayerName)
		{
			TestDriverRS testDriver = null;
			if (selectedLayerName.equalsIgnoreCase(getString(R.string.full_network_item_name)))
			{
				testDriver = new TestFullNetworkRS(m_rsContext);
			}

			String resultText = testDriver == null ? "No tests exist for this choice." : testDriver.executeTest();

			displayResults(resultText);
		}
	}

	/**
	 * Handles click on button to benchmark RenderScript implementation of the network.
	 */
	private class BenchRsBtnHandler implements View.OnClickListener
	{
		/**
		 * Spinner for choosing layer to test or benchmark.
		 */
		private final Spinner m_layerChoiceSpinner;

		/**
		 * Constructor.
		 * @param layerChoiceSpinner Spinner for choosing layer to test or benchmark.
		 */
		BenchRsBtnHandler(Spinner layerChoiceSpinner)
		{
			m_layerChoiceSpinner = layerChoiceSpinner;
		}

		/**
		 * Benchmarks RenderScript implementation of the network, for chosen layer.
		 * @param buttonView Button for benchmarking RenderScript implementation of the network.
		 */
		@Override
		public void onClick(View buttonView)
		{
			String selectedLayerName =  m_layerChoiceSpinner.getSelectedItem().toString();

			setupButtons(false);
			findViewById(R.id.resultText).setVisibility(View.GONE);
			findViewById(R.id.progressBar).setVisibility(View.VISIBLE);

			Thread thread = new Thread(() -> executeBenchmark(selectedLayerName));
			thread.start();
		}

		@WorkerThread
		private void executeBenchmark(String selectedLayerName)
		{
			BenchmarkDriverRS benchmarkDriver = null;
			if (selectedLayerName.equalsIgnoreCase(getString(R.string.convolutional_layer_name)))
			{
				benchmarkDriver = new BenchmarkConvolutionalLayerRS(m_rsContext);
			}
			else if (selectedLayerName.equalsIgnoreCase(getString(R.string.response_norm_layer_name)))
			{
				benchmarkDriver = new BenchmarkResponseNormalizationLayerRS(m_rsContext);
			}
			else if (selectedLayerName.equalsIgnoreCase(getString(R.string.max_pool_layer_name)))
			{
				benchmarkDriver = new BenchmarkMaxPoolLayerRS(m_rsContext);
			}
			else if (selectedLayerName.equalsIgnoreCase(getString(R.string.standard_layer_name)))
			{
				benchmarkDriver = new BenchmarkStandardLayerRS(m_rsContext);
			}
			else if (selectedLayerName.equalsIgnoreCase(getString(R.string.dropout_layer_name)))
			{
				benchmarkDriver = new BenchmarkDropoutLayerRS(m_rsContext);
			}
			else if (selectedLayerName.equalsIgnoreCase(getString(R.string.full_network_item_name)))
			{
				benchmarkDriver = new BenchmarkFullNetworkRS(m_rsContext);
			}

			String resultText = benchmarkDriver == null ? "No benchmarks exist for this choice." : benchmarkDriver.executeBenchmark();

			displayResults(resultText);
		}
	}

	private void setupButtons(boolean enabled)
	{
		findViewById(R.id.testRsBtn).setEnabled(enabled);
		findViewById(R.id.benchRsBtn).setEnabled(enabled);
		findViewById(R.id.layerChoice).setEnabled(enabled);
	}

	private void displayResults(final String resultText)
	{
		runOnUiThread(() -> {
			findViewById(R.id.progressBar).setVisibility(View.GONE);
			setupButtons(true);

			TextView resultTextView = findViewById(R.id.resultText);
			resultTextView.setText(Html.fromHtml(resultText));
			resultTextView.setVisibility(View.VISIBLE);
		});
	}
}
