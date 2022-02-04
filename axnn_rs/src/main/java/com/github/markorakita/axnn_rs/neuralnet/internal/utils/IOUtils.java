package com.github.markorakita.axnn_rs.neuralnet.internal.utils;

import androidx.annotation.RestrictTo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class IOUtils
{
    /**
     * Reads float buffer from stream.
     * @param bufferSize Buffer size to read.
     * @param stream Stream from which to read.
     * @param bigEndian Should stream be read in big endian.
     * @return Read float buffer.
     * @throws IOException Throws IOException in case when stream read fails.
     */
    @RestrictTo(RestrictTo.Scope.LIBRARY)
    public static float[] readFloatBufferFromStream(int bufferSize, InputStream stream, boolean bigEndian) throws IOException
    {
        byte[] bufferBytes = readByteBufferFromStream(bufferSize * 4, stream);

        FloatBuffer floatBuffer = ByteBuffer.wrap(bufferBytes).order(bigEndian ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] buffer = new float[bufferSize];
        floatBuffer.get(buffer);

        return buffer;
    }

    private static byte[] readByteBufferFromStream(int bufferSize, InputStream stream) throws IOException
    {
        byte[] buffer = new byte[bufferSize];
        int offset = 0;
        int numBytesRead;
        do
        {
            numBytesRead = stream.read(buffer, offset, bufferSize - offset);
            offset += numBytesRead;
        }
        while (numBytesRead > 0 && offset < bufferSize);

        if (offset < bufferSize)
        {
            throw new IOException("Not enough bytes in the stream.");
        }

        return buffer;
    }
}
