package com.github.markorakita.axnn_rs.neuralnet;

import android.graphics.Bitmap;
import android.renderscript.RenderScript;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import com.github.markorakita.axnn_rs.neuralnet.layers.ConvolutionalLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.FeaturesInputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.ImageInputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.InputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.LayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.OutputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.StandardLayerRS;

/**
 * Neural network with RenderScript implementation.
 */
public class NeuralNetRS
{
	/**
	 * Renderscript context.
	 */
    private final RenderScript m_rsContext;

	/**
	 * Network layers.
	 */
	private final ArrayList<LayerRS> m_layers;

	/**
	 * Constructor.
	 * @param rsContext Renderscript context.
	 */
	public NeuralNetRS(@NonNull RenderScript rsContext)
	{
		m_rsContext = rsContext;
		m_layers = new ArrayList<>();
	}

    /**
     * Adds layer to the network.
     * @param layer Layer to add.
     */
	public void addLayer(@NonNull LayerRS layer)
    {
        m_layers.add(layer);
    }

    /**
     * Removes layer from the network.
     * @param layerIndex Index of the layer in the network.
     * @return Returns true if layer was successfully deleted, false otherwise.
     */
    public boolean removeLayer(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= m_layers.size())
        {
            return false;
        }

        m_layers.remove(layerIndex);
        return true;
    }

    /**
     * Returns network layers.
     * @return Network layers.
     */
    @NonNull
    public ArrayList<LayerRS> getLayers()
    {
        return m_layers;
    }

    /**
     * Gets input layer of the network.
     * @return Input layer of the network.
     */
    @NonNull
    public InputLayerRS getInputLayer() throws Exception
    {
        if (m_layers.isEmpty() || !(m_layers.get(0) instanceof InputLayerRS))
        {
            throw new Exception("This network doesn't have input layer!");
        }

        return (InputLayerRS)m_layers.get(0);
    }

    /**
     * Gets layer of the network with given index.
     * @param layerIndex Layer index.
     * @return Layer of the network with given index.
     */
    public LayerRS getLayer(int layerIndex)
    {
        return m_layers.get(layerIndex);
    }

	/**
	 * Gets last layer in network.
	 * @return Last layer in network.
	 */
	public LayerRS getLastLayer()
	{
		return m_layers.get(m_layers.size() - 1);
	}

    /**
     * Gets output layer of the network.
     * @return Output layer of the network.
     * @throws Exception Throws generic exception in case when this network doesn't have an output layer.
     */
    public OutputLayerRS getOutputLayer() throws Exception
    {
        LayerRS lastLayer = getLastLayer();

        if (!(lastLayer instanceof OutputLayerRS))
        {
            throw new Exception("This network doesn't have output layer!");
        }

        return (OutputLayerRS)lastLayer;
    }

	/**
	 * Loads trained neural network model from stream.
	 * @param modelInputStream Model's input stream.
     * @param bigEndian Should we load parameters' values in big endian or small endian.
	 */
    @WorkerThread
	public void loadModel(@NonNull InputStream modelInputStream, boolean bigEndian) throws IOException
	{
		for (LayerRS layer: m_layers)
		{
			if (layer instanceof ConvolutionalLayerRS)
			{
				ConvolutionalLayerRS convLayer = (ConvolutionalLayerRS)layer;
				convLayer.loadFilters(modelInputStream, bigEndian);
				convLayer.loadBiases(modelInputStream, bigEndian);
			}
			else if (layer instanceof StandardLayerRS)
            {
                StandardLayerRS standardLayer = (StandardLayerRS)layer;
                standardLayer.loadWeights(modelInputStream, bigEndian);
                standardLayer.loadBiases(modelInputStream, bigEndian);
            }
		}
	}

    /**
     * Does forward propagation through the network.
     */
    @WorkerThread
    private void doForwardProp()
    {
        for (int i = 1; i < m_layers.size(); ++i)
        {
            LayerRS currentLayer = m_layers.get(i);
            currentLayer.setInputDataBuffer(m_layers.get(i - 1).getActivationDataBuffer());
            currentLayer.doForwardProp();

            m_rsContext.finish();
        }
    }

    /**
     * Classifies image.
     * @param image Image to classify.
     * @return Classification result.
     * @throws Exception Throws generic exception in case when classification fails.
     */
    @WorkerThread
    @NonNull
    public ClassificationResult classifyImage(@NonNull final Bitmap image) throws Exception
    {
        if (!(m_layers.get(0) instanceof ImageInputLayerRS))
        {
            throw new Exception("This network doesn't accept images as input data type.");
        }

        ImageInputLayerRS inputLayer = (ImageInputLayerRS)getInputLayer();
        if (!inputLayer.loadImage(image))
        {
            throw new Exception("Can't load input image.");
        }
        inputLayer.doForwardProp();

        doForwardProp();

        OutputLayerRS outputLayer = getOutputLayer();

        return new ClassificationResult(outputLayer.getPredictedClass(), outputLayer.getPredictionProbability());
    }

    /**
     * Classifies features.
     * @param features Features to classify.
     * @return Classification result.
     * @throws Exception Throws generic exception in case when classification fails.
     */
    @WorkerThread
    @NonNull
    public ClassificationResult classifyFeatures(@NonNull final float[] features) throws Exception
    {
        if (!(m_layers.get(0) instanceof FeaturesInputLayerRS))
        {
            throw new Exception("This component doesn't support features as input data type.");
        }

        FeaturesInputLayerRS inputLayer = (FeaturesInputLayerRS)getInputLayer();
        inputLayer.loadFeatures(features);
        inputLayer.doForwardProp();

        doForwardProp();

        OutputLayerRS outputLayer = getOutputLayer();

        return new ClassificationResult(outputLayer.getPredictedClass(), outputLayer.getPredictionProbability());
    }

    /**
     * Extracts features from the image, used for transfer learning.
     * @param image Image to extract features from.
     * @return Extracted features.
     * @throws Exception Throws generic exception in case when features extraction fails.
     */
    @WorkerThread
    @NonNull
    public float[] extractFeatures(@NonNull final Bitmap image) throws Exception
    {
        if (!(m_layers.get(0) instanceof ImageInputLayerRS))
        {
            throw new Exception("This network doesn't accept images as input data type.");
        }

        ImageInputLayerRS inputLayer = (ImageInputLayerRS)getInputLayer();
        if (!inputLayer.loadImage(image))
        {
            throw new Exception("Can't load input image.");
        }
        inputLayer.doForwardProp();

        doForwardProp();

        LayerRS lastLayer = getLastLayer();
        float[] featuresBuffer = new float[lastLayer.getActivationDataBufferSize()];
        lastLayer.getActivationDataBuffer().copyTo(featuresBuffer);

        return featuresBuffer;
    }
}
