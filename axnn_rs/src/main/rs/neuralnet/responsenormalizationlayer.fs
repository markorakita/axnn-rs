#pragma version(1)
#pragma rs java_package_name(com.github.markorakita.axnn_rs.neuralnet.layers)
#pragma rs_fp_relaxed

// Number of elements that we are packing per vector.
const int32_t c_numElPerVec = 4;

// Number of input data channels.
int32_t inputNumChannels;

// Input data width.
int32_t inputDataWidth;

// Input data height.
int32_t inputDataHeight;

// Input data buffer.
rs_allocation inputDataBuffer;

// Depth of normalization.
int32_t depth;

// Normalization bias.
float bias;

// Normalization alpha coefficient (see the formula in class header).
float alphaCoeff;

// Normalization beta coefficient (see the formula in class header).
float betaCoeff;

// Gets channel with specified index from channel vector.
// How can you make a numeric vector type without index accessor, wtf google, wtf...
static float getChannel(float4 channelVec, int32_t channelIndex)
{
	if (channelIndex == 0)
	{
		return channelVec.x;
	}
	else if (channelIndex == 1)
	{
		return channelVec.y;
	}
	else if (channelIndex == 2)
	{
		return channelVec.z;
	}
	else
	{
		return channelVec.w;
	}
}

// Does forward propagation through the layer.
float4 __attribute__((kernel)) propagate(uint32_t x)
{
 	// Initializing output  activations array.
	float activations[c_numElPerVec];
	for (int32_t i = 0; i < c_numElPerVec; ++i)
	{
	    activations[i] = 0.f;
	}

	const int32_t c_numChannelVecs = inputNumChannels / c_numElPerVec;
	const int32_t c_pixelIndex =  x / c_numChannelVecs;
	const int32_t c_pixelY = c_pixelIndex / inputDataWidth;
	const int32_t c_pixelX = c_pixelIndex % inputDataWidth;

	// Calculating first cross-channel sum.
	const int32_t c_channelVec = x % c_numChannelVecs;
	const int32_t c_inputChannel = c_channelVec * c_numElPerVec;
	const int32_t c_actualStartChannel = c_inputChannel - depth / 2;
	const int32_t c_startChannel = max(c_actualStartChannel, 0);
	const int32_t c_startChannelVec = c_startChannel / c_numElPerVec;
	const int32_t c_endChannel = min(c_actualStartChannel + depth, inputNumChannels);
	const int32_t c_endChannelVec = c_endChannel / c_numElPerVec;
	float crossChannelSum = 0.0f;
	float4 firstInputVec = rsGetElementAt_float4(inputDataBuffer, x - c_channelVec + c_startChannelVec);
	float4 lastInputVec = firstInputVec;
	int32_t firstInputVecFirstChannel = c_startChannel % c_numElPerVec;
	int32_t lastInputVecLastChannel = c_endChannel % c_numElPerVec;
	for (int32_t channel = firstInputVecFirstChannel; channel < min(c_endChannel - c_startChannelVec * c_numElPerVec, 4); ++channel)
	{
		float channelValue = getChannel(firstInputVec, channel);
		crossChannelSum += channelValue * channelValue;
	}
	if (c_endChannelVec > c_startChannelVec)
	{
		for (int32_t currChannelVec = c_startChannelVec + 1; currChannelVec < c_endChannelVec; ++currChannelVec)
		{
			float4 inputVec = rsGetElementAt_float4(inputDataBuffer, x - c_channelVec + currChannelVec);
			crossChannelSum += dot(inputVec, inputVec);
		}
		if (c_endChannel < inputNumChannels)
		{
			lastInputVec = rsGetElementAt_float4(inputDataBuffer, x - c_channelVec + c_endChannelVec);
			for (int32_t channel = 0; channel < lastInputVecLastChannel; ++channel)
			{
				float channelValue = getChannel(lastInputVec, channel);
				crossChannelSum += channelValue * channelValue;
			}
		}
	}
	activations[0] = native_powr(bias + alphaCoeff * crossChannelSum, -betaCoeff);

	// Calculating rest of cross-channel sums. Since we are only shifting by one channel,
	// to get the new sum we need to subtract first channel previously added and to add new end channel.
	for (int32_t inputChannel = 1; inputChannel < c_numElPerVec; ++inputChannel)
	{
		// Subtract first channel added to previous sum.
		if (c_actualStartChannel + inputChannel > 0)
		{
			float firstChannelValue = getChannel(firstInputVec, firstInputVecFirstChannel);
			crossChannelSum -= firstChannelValue * firstChannelValue;
			++firstInputVecFirstChannel;
			if (firstInputVecFirstChannel > 3)
			{
				firstInputVecFirstChannel = 0;
				firstInputVec = rsGetElementAt_float4(inputDataBuffer, x - c_channelVec + c_startChannelVec + 1);
			}
		}

		// Add next end channel.
		if (c_endChannel < inputNumChannels && lastInputVecLastChannel < 4)
		{
			float lastChannelValue = getChannel(lastInputVec, lastInputVecLastChannel);
			crossChannelSum += lastChannelValue * lastChannelValue;
			++lastInputVecLastChannel;
			if (lastInputVecLastChannel > 3 && c_endChannel < inputNumChannels - 4)
			{
				lastInputVecLastChannel = 0;
				lastInputVec = rsGetElementAt_float4(inputDataBuffer, x - c_channelVec + c_endChannelVec + 1);
			}
		}

		activations[inputChannel] = native_powr(bias + alphaCoeff * crossChannelSum, -betaCoeff);
	}

	float4 outputActivations = {activations[0], activations[1], activations[2], activations[3]};
	float4 inputActivations = rsGetElementAt_float4(inputDataBuffer, x);

	return inputActivations * outputActivations;
}