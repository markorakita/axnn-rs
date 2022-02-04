#pragma version(1)
#pragma rs java_package_name(com.github.markorakita.axnn_rs.neuralnet.layers)
#pragma rs_fp_relaxed

// Number of elements that we are packing per vector.
const int32_t c_numElPerVec = 4;

// Input data buffer.
rs_allocation inputDataBuffer;

// Input data size.
uint32_t inputDataSize;

// Weights buffer.
rs_allocation weightsBuffer;

// Biases buffer.
rs_allocation biasesBuffer;

// Calculates standard layer preactivations.
float __attribute__((kernel)) calculatePreactivations(uint32_t x)
{
	float preactivation = 0.f;

	// Positioning kernel.
	const uint32_t weightsOffset = x * inputDataSize;

	// Calculating preactivations.
	for (uint32_t inputIndex = 0; inputIndex < inputDataSize; ++inputIndex)
	{
        float inputData = rsGetElementAt_float(inputDataBuffer, inputIndex);
        float weight = rsGetElementAt_float(weightsBuffer, weightsOffset + inputIndex);

        preactivation += inputData * weight;
	}

    float bias = rsGetElementAt_float(biasesBuffer, x);

	return preactivation + bias;
}

// Calculates standard layer preactivations.
// Optimized to multiply four floats at a time.
float __attribute__((kernel)) calculatePreactivationsBy4(uint32_t x)
{
	float preactivation = 0.f;

	// Positioning kernel.
	const uint32_t numInputVecs = inputDataSize / c_numElPerVec;
	const uint32_t weightsOffset = x * numInputVecs;

	// Calculating preactivations.
	for (uint32_t inputIndex = 0; inputIndex < numInputVecs; ++inputIndex)
	{
        float4 inputVec = rsGetElementAt_float4(inputDataBuffer, inputIndex);
        float4 weightsVec = rsGetElementAt_float4(weightsBuffer, weightsOffset + inputIndex);

        preactivation += dot(inputVec, weightsVec);
	}

    float bias = rsGetElementAt_float(biasesBuffer, x);

	return preactivation + bias;
}