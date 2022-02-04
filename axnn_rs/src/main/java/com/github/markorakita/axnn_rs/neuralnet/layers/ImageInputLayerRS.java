package com.github.markorakita.axnn_rs.neuralnet.layers;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.renderscript.RenderScript;

import androidx.annotation.NonNull;

public class ImageInputLayerRS extends InputLayerRS
{
    /**
     * Input layer works with RGB format images, but we need number of channels to be divisible by four
     * for easier calculation. We fill last channel with zeroes.
     */
    private static final int c_inputNumChannels = 4;

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     * @param imageWidth Input image width.
     * @param imageHeight Input image height.
     */
    public ImageInputLayerRS(@NonNull RenderScript rsContext, int imageWidth, int imageHeight)
    {
        super(rsContext, imageWidth, imageHeight, c_inputNumChannels);
    }

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     * @param imageWidth Input image width.
     * @param imageHeight Input image height.
     * @param inputDataMeans Mean per channel on which to normalize input data.
     * @param inputDataStDevs Standard deviation per channel on which to normalize input data.
     */
    public ImageInputLayerRS(@NonNull RenderScript rsContext, int imageWidth, int imageHeight, float[] inputDataMeans, float[] inputDataStDevs)
    {
        super(rsContext, imageWidth, imageHeight, c_inputNumChannels, inputDataMeans, inputDataStDevs);
    }

    /**
     * Loads input image.
     * @param image Input image.
     * @return True if image is successfully loaded, false otherwise.
     */
    public boolean loadImage(@NonNull Bitmap image)
    {
        // Calculating dimensions for resize, so that resized image fits into rectangle defined by m_inputDataHeight and m_inputDataWidth.
        int resizedWidth, resizedHeight;
        if (image.getWidth() > image.getHeight())
        {
            resizedWidth = m_inputDataHeight * image.getWidth() / image.getHeight();
            resizedHeight = m_inputDataHeight;
        }
        else
        {
            resizedWidth = m_inputDataWidth;
            resizedHeight = m_inputDataWidth * image.getHeight() / image.getWidth();
        }

        // Resizing image.
        Bitmap resizedImage = Bitmap.createScaledBitmap(image, resizedWidth, resizedHeight, true);

        // Extracting resized image pixels.
        int[] resizedImagePixels = new int[resizedWidth * resizedHeight];
        resizedImage.getPixels(resizedImagePixels, 0, resizedWidth, 0, 0, resizedWidth, resizedHeight);
        // We don't need resized image anymore.
        resizedImage.recycle();

        // Cropping input image center patch from resized image.
        m_unnormalizedInputDataBuffer = new float[m_activationDataBufferSize];
        int cropX = (resizedWidth - m_inputDataWidth) / 2;
        int cropY = (resizedHeight - m_inputDataHeight) / 2;
        int endRow = cropY + m_inputDataHeight;
        int endCol = cropX + m_inputDataWidth;
        for (int row = cropY; row < endRow; ++row)
        {
            int rowOffset = (row - cropY) * m_inputDataWidth * m_inputDataNumChannels;
            for (int col = cropX; col < endCol; ++col)
            {
                int colOffset = rowOffset + (col - cropX) * m_inputDataNumChannels;
                int pixel = resizedImagePixels[row * resizedWidth + col];
                m_unnormalizedInputDataBuffer[colOffset] = Color.red(pixel);
                m_unnormalizedInputDataBuffer[colOffset + 1] = Color.green(pixel);
                m_unnormalizedInputDataBuffer[colOffset + 2] = Color.blue(pixel);
                m_unnormalizedInputDataBuffer[colOffset + 3] = 0.f;
            }
        }

        return true;
    }
}
