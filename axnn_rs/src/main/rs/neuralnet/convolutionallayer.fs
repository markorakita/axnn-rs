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

// Number of convolutional filters.
int32_t numFilters;

// Width of a filter.
int32_t filterWidth;

// Height of a filter.
int32_t filterHeight;

// Filters buffer.
rs_allocation filtersBuffer;

// Biases buffer.
rs_allocation biasesBuffer;

// Padding in dimension X.
int32_t paddingX;

// Padding in dimension Y.
int32_t paddingY;

// Stride for patching.
int32_t stride;

// Number of patches to apply filters on in dimension X.
int32_t numPatchesX;

// Number of patches to apply filters on in dimension Y.
int32_t numPatchesY;

// Does forward propagation through the layer.
float4 __attribute__((kernel)) propagate(uint32_t x)
{
    // Initializing output  activations array.
	float activations[c_numElPerVec];
	for (int32_t i = 0; i < c_numElPerVec; ++i)
	{
	    activations[i] = 0.f;
	}

	// Positioning kernel.
	const int32_t c_numChannelVecs = inputNumChannels / c_numElPerVec;
	const int32_t c_numFilterPixels = filterWidth * filterHeight * c_numChannelVecs;
	int32_t patchIndex = c_numElPerVec * x / numFilters;
	int32_t patchOffsetX = -paddingX + (patchIndex % numPatchesX) * stride;
	int32_t patchOffsetY = -paddingY + (patchIndex / numPatchesX) * stride;
	int32_t filterOffset = x % (numFilters / c_numElPerVec);

    // Calculating convolution for one patch and number of filters.
	for (int32_t filterPixelY = 0; filterPixelY < filterHeight; ++filterPixelY)
	{
		int32_t imagePixelY = patchOffsetY + filterPixelY;
		if (imagePixelY >= 0 && imagePixelY < inputDataHeight)
		{
			int32_t imagePixelYOffset = imagePixelY * inputDataWidth * c_numChannelVecs;
			int32_t filterPixelYOffset = filterPixelY * filterWidth * c_numChannelVecs;
			for (int32_t filterPixelX = 0; filterPixelX < filterWidth; ++filterPixelX)
			{
				int32_t imagePixelX = patchOffsetX + filterPixelX;
				if (imagePixelX >= 0 && imagePixelX < inputDataWidth)
				{
					int32_t imagePixelXOffset = imagePixelYOffset + imagePixelX * c_numChannelVecs;
					int32_t filterPixelXOffset = filterPixelYOffset + filterPixelX * c_numChannelVecs;
					for (int32_t channelIndex = 0; channelIndex < c_numChannelVecs; ++channelIndex)
					{
						float4 imagePixels = rsGetElementAt_float4(inputDataBuffer, imagePixelXOffset + channelIndex);
						for (int32_t filterIndex = 0; filterIndex < c_numElPerVec; ++filterIndex)
						{
							float4 filterPixels = rsGetElementAt_float4(filtersBuffer, (filterOffset * c_numElPerVec + filterIndex) * c_numFilterPixels +
								filterPixelXOffset + channelIndex);
							activations[filterIndex] += dot(imagePixels, filterPixels);
						}
					}
				}
			}
		}
	}

	float4 outputActivations = {activations[0], activations[1], activations[2], activations[3]};
	float4 biases = rsGetElementAt_float4(biasesBuffer, filterOffset);

	return outputActivations + biases;
}