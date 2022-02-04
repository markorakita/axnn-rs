# AXNN

AXNN is an open source Android SDK for running on-device inference on neural network models trained with [XNN](/../../../xnn) framework. It provides hardware acceleration by using [RenderScript](https://developer.android.com/guide/topics/renderscript/compute) Android framework in order to run inference on device GPU, or parallelized across multiple CPU cores if GPU is not available.

You can find [here](/../../../xnn/tree/master/Docs/Supported%20layers.md) the list of all supported neural network layers.

#### 🚨 CAUTION: 🚨 Starting with Android 12 the RenderScript APIs are deprecated. They will continue to function, but device and component manufacturers will stop providing hardware acceleration support over time. I plan on releasing new AXNN SDK which uses OpenGL for GPU acceleration.

## Setup

Make sure Maven Central repository is added in your project `build.gradle` file:
```gradle
buildscript {
    repositories {
        mavenCentral()
    }
}

allprojects {
    repositories {
        mavenCentral()
    }
}
```

Include AXNN in your application `build.gradle` file:
```gradle
dependencies {
    implementation 'com.github.markorakita:axnn-rs:1.0.0'
}
```

## Getting Started

You can use NeuralNetFactory class to create some of the well known neural network architectures:
```java
RenderScript rsContext = RenderScript.create(getApplicationContext());

NeuralNetRS neuralNet = NeuralNetFactory.createAlexNetDnn(m_rsContext);
```

Or you can create neural network from scratch:
```java
RenderScript rsContext = RenderScript.create(getApplicationContext());

NeuralNetRS neuralNet = new NeuralNetRS(rsContext);
    
InputLayerRS inputLayer = new FeaturesInputLayerRS(rsContext, /*number of features:*/ 40);
neuralNet.addLayer(inputLayer);
    
StandardLayerRS standardLayer1 = new StandardLayerRS(rsContext, inputLayer.getActivationNumChannels(),
        inputLayer.getActivationDataWidth(), inputLayer.getActivationDataHeight(),
        /*number of neurons:*/ 128, ActivationFunction.ActivationFunctionType.ReLU);
neuralNet.addLayer(standardLayer1);
    
StandardLayerRS standardLayer2 = new StandardLayerRS(rsContext, standardLayer1.getActivationNumChannels(),
        standardLayer1.getActivationDataWidth(), standardLayer1.getActivationDataHeight(),
        /*number of neurons:*/ 10, ActivationFunction.ActivationFunctionType.Linear);
neuralNet.addLayer(standardLayer2);
    
SoftMaxLayerRS softMaxLayer = new SoftMaxLayerRS(rsContext, standardLayer2.getActivationDataBufferSize());
neuralNet.addLayer(softMaxLayer);
    
OutputLayerRS outputLayer = new OutputLayerRS(softMaxLayer.getActivationDataBufferSize(),
        OutputLayerRS.LossFunctionType.CrossEntropy);
neuralNet.addLayer(outputLayer);
```

Before you can use the created neural network for inference, you need to load trained XNN model ([converted](/../../../xnn/tree/master/Docs/Command%20line%20parameters.md#model-conversion) into AXNN format):
```java
try (InputStream modelInputStream = rsContext.getApplicationContext().getAssets().open("model.xnnm"))
{
    // This can't be called from UI thread!
    neuralNet.loadModel(modelInputStream, true);
}
catch (IOException exc)
{
    // Handle exception...
}
```

You can then use network for inference by calling some of these functions:
```java
try
{
    // These can't be called from UI thread!
    ClassificationResult imageResult = neuralNet.classifyImage(image); // Bitmap image
    ClassificationResult featuresResult = neuralNet.classifyFeatures(features); // float[] features
}
catch (Exception exc)
{
    // Handle exception...
}
```

You can also use network to extract features, which can then be fed into some other network for classification (used for transfer learning models):
```java
try
{
    // These can't be called from UI thread!
    float[] imageFeatures = featuresNeuralNet.extractFeatures(image); // Bitmap image
    ClassificationResult result = classificationNeuralNet.classifyFeatures(imageFeatures);
}
catch (Exception exc)
{
    // Handle exception...
}
```
