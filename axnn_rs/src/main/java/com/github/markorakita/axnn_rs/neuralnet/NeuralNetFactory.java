package com.github.markorakita.axnn_rs.neuralnet;

import android.renderscript.RenderScript;

import androidx.annotation.NonNull;

import com.github.markorakita.axnn_rs.neuralnet.layers.ConvolutionalLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.DropoutLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.ImageInputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.InputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.MaxPoolLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.OutputLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.ResponseNormalizationLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.SoftMaxLayerRS;
import com.github.markorakita.axnn_rs.neuralnet.layers.StandardLayerRS;

/**
 * Factory for creating some of the popular neural nets.
 */
public class NeuralNetFactory
{
    /**
     * Creates AlexNet deep neural network.
     */
    @NonNull
    public static NeuralNetRS createAlexNetDnn(@NonNull RenderScript rsContext)
    {
        NeuralNetRS alexNet = new NeuralNetRS(rsContext);

        //------------------------------------------|
        //		Input layer
        //------------------------------------------|
        final int l1_inputDataWidth = 224;
        final int l1_inputDataHeight = 224;
        final float[] l1_inputDataMeans = {123.f, 116.f, 103.f};
        final float[] l1_inputDataStDevs = {1.f, 1.f, 1.f};
        InputLayerRS inputLayer = new ImageInputLayerRS(rsContext, l1_inputDataWidth, l1_inputDataHeight, l1_inputDataMeans, l1_inputDataStDevs);
        alexNet.addLayer(inputLayer);

        //------------------------------------------|
        //		Convolutional layer 1
        //------------------------------------------|
        final int l2_numFilters = 64;
        final int l2_filterWidth = 11;
        final int l2_filterHeight = 11;
        final int l2_paddingX = 0;
        final int l2_paddingY = 0;
        final int l2_stride = 4;
        final ActivationFunction.ActivationFunctionType l2_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        ConvolutionalLayerRS convLayer1 = new ConvolutionalLayerRS(rsContext, inputLayer.getActivationNumChannels(), inputLayer.getActivationDataWidth(),
                inputLayer.getActivationDataHeight(), l2_numFilters, l2_filterWidth, l2_filterHeight, l2_paddingX, l2_paddingY, l2_stride,
                l2_activationFunction);
        alexNet.addLayer(convLayer1);

        //------------------------------------------|
        //		ResponseNormalization layer 1
        //------------------------------------------|
        final int l3_depth = 5;
        final float l3_bias = 2;
        final float l3_alphaCoeff = 0.0001f;
        final float l3_betaCoeff = 0.75f;
        ResponseNormalizationLayerRS reNormLayer1 = new ResponseNormalizationLayerRS(rsContext, convLayer1.getActivationNumChannels(),
                convLayer1.getActivationDataWidth(), convLayer1.getActivationDataHeight(), l3_depth, l3_bias, l3_alphaCoeff, l3_betaCoeff);
        alexNet.addLayer(reNormLayer1);

        //------------------------------------------|
        //		MaxPool layer 1
        //------------------------------------------|
        final int l4_unitWidth = 3;
        final int l4_unitHeight = 3;
        final int l4_paddingX = 0;
        final int l4_paddingY =0;
        final int l4_unitStride = 2;
        MaxPoolLayerRS maxPoolLayer1 = new MaxPoolLayerRS(rsContext, reNormLayer1.getActivationNumChannels(), reNormLayer1.getActivationDataWidth(),
                reNormLayer1.getActivationDataHeight(), l4_unitWidth, l4_unitHeight, l4_paddingX, l4_paddingY, l4_unitStride);
        alexNet.addLayer(maxPoolLayer1);

        //------------------------------------------|
        //		Convolutional layer 2
        //------------------------------------------|
        final int l5_numFilters = 192;
        final int l5_filterWidth = 5;
        final int l5_filterHeight = 5;
        final int l5_paddingX = 2;
        final int l5_paddingY = 2;
        final int l5_stride = 1;
        final ActivationFunction.ActivationFunctionType l5_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        ConvolutionalLayerRS convLayer2 = new ConvolutionalLayerRS(rsContext, maxPoolLayer1.getActivationNumChannels(), maxPoolLayer1.getActivationDataWidth(),
                maxPoolLayer1.getActivationDataHeight(), l5_numFilters, l5_filterWidth, l5_filterHeight, l5_paddingX, l5_paddingY, l5_stride,
                l5_activationFunction);
        alexNet.addLayer(convLayer2);

        //------------------------------------------|
        //		ResponseNormalization layer 2
        //------------------------------------------|
        final int l6_depth = 5;
        final float l6_bias = 2;
        final float l6_alphaCoeff = 0.0001f;
        final float l6_betaCoeff = 0.75f;
        ResponseNormalizationLayerRS reNormLayer2 = new ResponseNormalizationLayerRS(rsContext, convLayer2.getActivationNumChannels(),
                convLayer2.getActivationDataWidth(), convLayer2.getActivationDataHeight(), l6_depth, l6_bias, l6_alphaCoeff, l6_betaCoeff);
        alexNet.addLayer(reNormLayer2);

        //------------------------------------------|
        //		MaxPool layer 2
        //------------------------------------------|
        final int l7_unitWidth = 3;
        final int l7_unitHeight = 3;
        final int l7_paddingX = 0;
        final int l7_paddingY =0;
        final int l7_unitStride = 2;
        MaxPoolLayerRS maxPoolLayer2 = new MaxPoolLayerRS(rsContext, reNormLayer2.getActivationNumChannels(), reNormLayer2.getActivationDataWidth(),
                reNormLayer2.getActivationDataHeight(), l7_unitWidth, l7_unitHeight, l7_paddingX, l7_paddingY, l7_unitStride);
        alexNet.addLayer(maxPoolLayer2);

        //------------------------------------------|
        //		Convolutional layer 3
        //------------------------------------------|
        final int l8_numFilters = 384;
        final int l8_filterWidth = 3;
        final int l8_filterHeight = 3;
        final int l8_paddingX = 1;
        final int l8_paddingY = 1;
        final int l8_stride = 1;
        final ActivationFunction.ActivationFunctionType l8_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        ConvolutionalLayerRS convLayer3 = new ConvolutionalLayerRS(rsContext, maxPoolLayer2.getActivationNumChannels(), maxPoolLayer2.getActivationDataWidth(),
                maxPoolLayer2.getActivationDataHeight(), l8_numFilters, l8_filterWidth, l8_filterHeight, l8_paddingX, l8_paddingY, l8_stride,
                l8_activationFunction);
        alexNet.addLayer(convLayer3);

        //------------------------------------------|
        //		Convolutional layer 4
        //------------------------------------------|
        final int l9_numFilters = 256;
        final int l9_filterWidth = 3;
        final int l9_filterHeight = 3;
        final int l9_paddingX = 1;
        final int l9_paddingY = 1;
        final int l9_stride = 1;
        final ActivationFunction.ActivationFunctionType l9_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        ConvolutionalLayerRS convLayer4 = new ConvolutionalLayerRS(rsContext, convLayer3.getActivationNumChannels(), convLayer3.getActivationDataWidth(),
                convLayer3.getActivationDataHeight(), l9_numFilters, l9_filterWidth, l9_filterHeight, l9_paddingX, l9_paddingY, l9_stride,
                l9_activationFunction);
        alexNet.addLayer(convLayer4);

        //------------------------------------------|
        //		Convolutional layer 5
        //------------------------------------------|
        final int l10_numFilters = 256;
        final int l10_filterWidth = 3;
        final int l10_filterHeight = 3;
        final int l10_paddingX = 1;
        final int l10_paddingY = 1;
        final int l10_stride = 1;
        final ActivationFunction.ActivationFunctionType l10_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        ConvolutionalLayerRS convLayer5 = new ConvolutionalLayerRS(rsContext, convLayer4.getActivationNumChannels(), convLayer4.getActivationDataWidth(),
                convLayer4.getActivationDataHeight(), l10_numFilters, l10_filterWidth, l10_filterHeight, l10_paddingX, l10_paddingY, l10_stride,
                l10_activationFunction);
        alexNet.addLayer(convLayer5);

        //------------------------------------------|
        //		MaxPool layer 3
        //------------------------------------------|
        final int l11_unitWidth = 3;
        final int l11_unitHeight = 3;
        final int l11_paddingX = 0;
        final int l11_paddingY =0;
        final int l11_unitStride = 2;
        MaxPoolLayerRS maxPoolLayer3 = new MaxPoolLayerRS(rsContext, convLayer5.getActivationNumChannels(), convLayer5.getActivationDataWidth(),
                convLayer5.getActivationDataHeight(), l11_unitWidth, l11_unitHeight, l11_paddingX, l11_paddingY, l11_unitStride);
        alexNet.addLayer(maxPoolLayer3);

        //------------------------------------------|
        //		Standard layer 1
        //------------------------------------------|
        final int l12_numNeurons = 4096;
        final ActivationFunction.ActivationFunctionType l12_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        StandardLayerRS standardLayer1 = new StandardLayerRS(rsContext, maxPoolLayer3.getActivationNumChannels(), maxPoolLayer3.getActivationDataWidth(),
                maxPoolLayer3.getActivationDataHeight(), l12_numNeurons, l12_activationFunction);
        alexNet.addLayer(standardLayer1);

        //------------------------------------------|
        //		Dropout layer 1
        //------------------------------------------|
        final float l13_dropProbability = 0.5f;
        DropoutLayerRS dropoutLayer1 = new DropoutLayerRS(rsContext, standardLayer1.getActivationNumChannels(), standardLayer1.getActivationDataWidth(),
                standardLayer1.getActivationDataHeight(), l13_dropProbability);
        alexNet.addLayer(dropoutLayer1);

        //------------------------------------------|
        //		Standard layer 2
        //------------------------------------------|
        final int l14_numNeurons = 4096;
        final ActivationFunction.ActivationFunctionType l14_activationFunction = ActivationFunction.ActivationFunctionType.ReLU;
        StandardLayerRS standardLayer2 = new StandardLayerRS(rsContext, dropoutLayer1.getActivationNumChannels(), dropoutLayer1.getActivationDataWidth(),
                dropoutLayer1.getActivationDataHeight(), l14_numNeurons, l14_activationFunction);
        alexNet.addLayer(standardLayer2);

        //------------------------------------------|
        //		Dropout layer 2
        //------------------------------------------|
        final float l15_dropProbability = 0.5f;
        DropoutLayerRS dropoutLayer2 = new DropoutLayerRS(rsContext, standardLayer2.getActivationNumChannels(), standardLayer2.getActivationDataWidth(),
                standardLayer2.getActivationDataHeight(), l15_dropProbability);
        alexNet.addLayer(dropoutLayer2);

        //------------------------------------------|
        //		Standard layer 3
        //------------------------------------------|
        final int l16_numNeurons = 1000;
        final ActivationFunction.ActivationFunctionType l16_activationFunction = ActivationFunction.ActivationFunctionType.Linear;
        StandardLayerRS standardLayer3 = new StandardLayerRS(rsContext, dropoutLayer2.getActivationNumChannels(), dropoutLayer2.getActivationDataWidth(),
                dropoutLayer2.getActivationDataHeight(), l16_numNeurons, l16_activationFunction);
        alexNet.addLayer(standardLayer3);

        //------------------------------------------|
        //		SoftMax layer
        //------------------------------------------|
        SoftMaxLayerRS softMaxLayer = new SoftMaxLayerRS(rsContext, standardLayer3.getActivationDataBufferSize());
        alexNet.addLayer(softMaxLayer);

        //------------------------------------------|
        //		Output layer
        //------------------------------------------|
        OutputLayerRS outputLayer = new OutputLayerRS(softMaxLayer.getActivationDataBufferSize(), OutputLayerRS.LossFunctionType.CrossEntropy);
        alexNet.addLayer(outputLayer);

        return alexNet;
    }
}
