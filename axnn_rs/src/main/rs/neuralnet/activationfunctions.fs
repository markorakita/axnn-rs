#pragma version(1)
#pragma rs java_package_name(com.github.markorakita.axnn_rs.neuralnet)
#pragma rs_fp_relaxed

// Preactivation data buffer.
rs_allocation preactivationDataBuffer;

// Activation data buffer.
rs_allocation activationDataBuffer;

// Activation data gradients buffer.
rs_allocation activationDataGradientsBuffer;

// Activation alpha parameter.
float activationAlpha;

// Applies ReLU activation.
float __attribute__((kernel)) applyReLUActivation(uint32_t x)
{
    float preactivationData = rsGetElementAt_float(preactivationDataBuffer, x);

	return fmax(preactivationData, 0.f);
}

// Applies ReLU activation.
// Optimized to process four floats at a time.
float4 __attribute__((kernel)) applyReLUActivationBy4(uint32_t x)
{
    float4 preactivationData = rsGetElementAt_float4(preactivationDataBuffer, x);
    float4 zeroVec = {0.f, 0.f, 0.f, 0.f};

	return fmax(preactivationData, zeroVec);
}

// Calculates ReLU preactivation gradients.
float __attribute__((kernel)) calculateReLUPreactivationGradients(uint32_t x)
{
    float activationData = rsGetElementAt_float(activationDataBuffer, x);
    float activationDataGradient = rsGetElementAt_float(activationDataGradientsBuffer, x);

    return activationData > 0.f ? activationDataGradient : 0.f;
}

// Applies ELU activation.
float __attribute__((kernel)) applyELUActivation(uint32_t x)
{
    float preactivationData = rsGetElementAt_float(preactivationDataBuffer, x);

	return preactivationData >= 0.f ? preactivationData : activationAlpha * (native_exp(preactivationData) - 1.f);
}

// Applies ELU activation.
// Optimized to process four floats at a time.
float4 __attribute__((kernel)) applyELUActivationBy4(uint32_t x)
{
    float4 preactivationData = rsGetElementAt_float4(preactivationDataBuffer, x);
    float4 zeroVec = {0.f, 0.f, 0.f, 0.f};

	return fmax(preactivationData, zeroVec) + fmin(zeroVec, activationAlpha * (native_exp(preactivationData) - 1.f));
}

// Calculates ELU preactivation gradients.
float __attribute__((kernel)) calculateELUPreactivationGradients(uint32_t x)
{
    float activationData = rsGetElementAt_float(activationDataBuffer, x);
    float activationDataGradient = rsGetElementAt_float(activationDataGradientsBuffer, x);

    return activationData > 0.f ? activationDataGradient : activationDataGradient * (activationData + activationAlpha);
}

// Applies LeakyReLU activation.
float __attribute__((kernel)) applyLeakyReLUActivation(uint32_t x)
{
    float preactivationData = rsGetElementAt_float(preactivationDataBuffer, x);

	return preactivationData >= 0.f ? preactivationData : activationAlpha * preactivationData;
}

// Applies LeakyReLU activation.
// Optimized to process four floats at a time.
float4 __attribute__((kernel)) applyLeakyReLUActivationBy4(uint32_t x)
{
    float4 preactivationData = rsGetElementAt_float4(preactivationDataBuffer, x);
    float4 zeroVec = {0.f, 0.f, 0.f, 0.f};

	return fmax(preactivationData, zeroVec) + fmin(zeroVec, activationAlpha * preactivationData);
}

// Calculates LeakyReLU preactivation gradients.
float __attribute__((kernel)) calculateLeakyReLUPreactivationGradients(uint32_t x)
{
    float activationData = rsGetElementAt_float(activationDataBuffer, x);
    float activationDataGradient = rsGetElementAt_float(activationDataGradientsBuffer, x);

    return activationData > 0.f ? activationDataGradient : activationDataGradient * activationAlpha;
}

// Applies Sigmoid activation.
float __attribute__((kernel)) applySigmoidActivation(uint32_t x)
{
    float preactivationData = rsGetElementAt_float(preactivationDataBuffer, x);

	return preactivationData >= 0.f ?
	    native_recip(1.f + native_exp(-preactivationData)) :
	    (1.f - native_recip(1.f + native_exp(preactivationData)));
}

// Applies Sigmoid activation.
// Optimized to process four floats at a time.
float4 __attribute__((kernel)) applySigmoidActivationBy4(uint32_t x)
{
    float4 preactivationData = rsGetElementAt_float4(preactivationDataBuffer, x);

    float4 activationData = {
        preactivationData.x >= 0.f ? native_recip(1.f + native_exp(-preactivationData.x)) : (1.f - native_recip(1.f + native_exp(preactivationData.x))),
        preactivationData.y >= 0.f ? native_recip(1.f + native_exp(-preactivationData.y)) : (1.f - native_recip(1.f + native_exp(preactivationData.y))),
        preactivationData.z >= 0.f ? native_recip(1.f + native_exp(-preactivationData.z)) : (1.f - native_recip(1.f + native_exp(preactivationData.z))),
        preactivationData.w >= 0.f ? native_recip(1.f + native_exp(-preactivationData.w)) : (1.f - native_recip(1.f + native_exp(preactivationData.w)))
    };

    return activationData;
}

// Calculates Sigmoid preactivation gradients.
float __attribute__((kernel)) calculateSigmoidPreactivationGradients(uint32_t x)
{
    float activationData = rsGetElementAt_float(activationDataBuffer, x);
    float activationDataGradient = rsGetElementAt_float(activationDataGradientsBuffer, x);

    return activationDataGradient * activationData * (1.f - activationData);
}

// Applies Tanh activation.
float __attribute__((kernel)) applyTanhActivation(uint32_t x)
{
    float preactivationData = rsGetElementAt_float(preactivationDataBuffer, x);

	return preactivationData >= 0.f ?
	    (2.f * native_recip(1.f + native_exp(-2.f * preactivationData)) - 1.f) :
	    (1.f - 2.f * native_recip(1.f + native_exp(2.f * preactivationData)));
}

// Applies Tanh activation.
// Optimized to process four floats at a time.
float4 __attribute__((kernel)) applyTanhActivationBy4(uint32_t x)
{
    float4 preactivationData = rsGetElementAt_float4(preactivationDataBuffer, x);

    float4 activationData = {
            preactivationData.x >= 0.f ? (2.f * native_recip(1.f + native_exp(-2.f * preactivationData.x)) - 1.f) : (1.f - 2.f * native_recip(1.f + native_exp(2.f * preactivationData.x))),
            preactivationData.y >= 0.f ? (2.f * native_recip(1.f + native_exp(-2.f * preactivationData.y)) - 1.f) : (1.f - 2.f * native_recip(1.f + native_exp(2.f * preactivationData.y))),
            preactivationData.z >= 0.f ? (2.f * native_recip(1.f + native_exp(-2.f * preactivationData.z)) - 1.f) : (1.f - 2.f * native_recip(1.f + native_exp(2.f * preactivationData.z))),
            preactivationData.w >= 0.f ? (2.f * native_recip(1.f + native_exp(-2.f * preactivationData.w)) - 1.f) : (1.f - 2.f * native_recip(1.f + native_exp(2.f * preactivationData.w)))
        };

    return activationData;
}

// Calculates Tanh preactivation gradients.
float __attribute__((kernel)) calculateTanhPreactivationGradients(uint32_t x)
{
    float activationData = rsGetElementAt_float(activationDataBuffer, x);
    float activationDataGradient = rsGetElementAt_float(activationDataGradientsBuffer, x);

    return activationDataGradient * (1.f - activationData * activationData);
}