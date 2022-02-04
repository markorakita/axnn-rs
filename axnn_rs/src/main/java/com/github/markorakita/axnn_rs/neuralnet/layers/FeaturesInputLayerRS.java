package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.renderscript.RenderScript;

import androidx.annotation.NonNull;

public class FeaturesInputLayerRS extends InputLayerRS
{
    /**
     * Constructor.
     * @param rsContext Renderscript context.
     * @param numberOfFeatures Number of features.
     */
    public FeaturesInputLayerRS(@NonNull RenderScript rsContext, int numberOfFeatures)
    {
        super(rsContext, numberOfFeatures, 1, 1);
    }

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     * @param numberOfFeatures Number of features.
     * @param featuresMean Mean on which to normalize features.
     * @param featuresStandardDeviation Standard deviation on which to normalize features.
     */
    public FeaturesInputLayerRS(@NonNull RenderScript rsContext, int numberOfFeatures, float featuresMean, float featuresStandardDeviation)
    {
        super(rsContext, numberOfFeatures, 1, 1, new float[]{featuresMean}, new float[]{featuresStandardDeviation});
    }

    /**
     * Loads input features.
     * @param features Input features.
     */
    public void loadFeatures(@NonNull float[] features)
    {
        m_unnormalizedInputDataBuffer = features;
    }
}
