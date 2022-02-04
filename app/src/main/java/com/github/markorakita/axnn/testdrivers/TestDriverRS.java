package com.github.markorakita.axnn.testdrivers;

import android.renderscript.RenderScript;

public abstract class TestDriverRS
{
    /**
     * RenderScript context.
     */
    protected final RenderScript m_rsContext;

    /**
     * Constructor.
     * @param rsContext Renderscript context.
     */
    protected TestDriverRS(RenderScript rsContext)
    {
        m_rsContext = rsContext;
    }

    public abstract String executeTest();
}
