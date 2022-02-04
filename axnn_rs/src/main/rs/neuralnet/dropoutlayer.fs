#pragma version(1)
#pragma rs java_package_name(com.github.markorakita.axnn_rs.neuralnet.layers)
#pragma rs_fp_relaxed

// Number of elements that we are packing per vector.
const int32_t c_numElPerVec = 4;

// Input data buffer.
rs_allocation inputDataBuffer;

float dropProbability;

// Does forward propagation through the layer.
// Optimized to process four floats at a time.
float4 __attribute__((kernel)) propagate(uint32_t x)
{
    float4 inputData = rsGetElementAt_float4(inputDataBuffer, x);

    // During inference dropout layer simply passes through the input scaled to match probability that it wont be dropped.
	return inputData * (1.f - dropProbability);
}