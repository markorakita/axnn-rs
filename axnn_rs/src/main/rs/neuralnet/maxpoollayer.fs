#pragma version(1)
#pragma rs java_package_name(com.github.markorakita.axnn_rs.neuralnet.layers)
#pragma rs_fp_relaxed

// Number of elements that we are packing per vector.
const int32_t c_numElPerVec = 4;

// Max float valule.
const float FLT_MAX = 3.402823466e+38F;

// Number of input data channels.
int32_t inputNumChannels;

// Input data width.
int32_t inputDataWidth;

// Input data height.
int32_t inputDataHeight;

// Input data buffer.
rs_allocation inputDataBuffer;

// Width of the pooling unit.
int unitWidth;

// Height of the pooling unit.
int unitHeight;

// Padding in dimension X.
int paddingX;

// Padding in dimension Y.
int paddingY;

// Stride of the pooling unit.
int unitStride;

// Number of pooling units in dimension X.
int numUnitsX;

// Does forward propagation through the layer.
float4 __attribute__((kernel)) propagate(uint32_t x)
{
	// Initializing output  activations
	float4 activations = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

	// Positioning kernel.
	const int32_t c_numChannelVecs = inputNumChannels / c_numElPerVec;
	const int32_t c_unitIndex = x / c_numChannelVecs;
	const int32_t c_channelVec = x % c_numChannelVecs;
	const int32_t c_unitOffsetX = -paddingX + (c_unitIndex % numUnitsX) * unitStride;
    const int32_t c_unitOffsetY = -paddingY + (c_unitIndex / numUnitsX) * unitStride;

    // Calculating max for one unit and c_numElPerVec channels.
    for (int32_t unitPixelY = 0; unitPixelY < unitHeight; ++unitPixelY)
	{
		int32_t imagePixelY = c_unitOffsetY + unitPixelY;
		if (imagePixelY >= 0 && imagePixelY < inputDataHeight)
		{
			int32_t imagePixelYOffset = imagePixelY * inputDataWidth * c_numChannelVecs;
			for (int32_t unitPixelX = 0; unitPixelX < unitWidth; ++unitPixelX)
			{
				int32_t imagePixelX = c_unitOffsetX + unitPixelX;
				if (imagePixelX >= 0 && imagePixelX < inputDataWidth)
				{
					int32_t imagePixelXOffset = imagePixelYOffset + imagePixelX * c_numChannelVecs;
					float4 imagePixels = rsGetElementAt_float4(inputDataBuffer, imagePixelXOffset + c_channelVec);
					activations = max(activations, imagePixels);
				}
			}
		}
	}

	return activations;
}