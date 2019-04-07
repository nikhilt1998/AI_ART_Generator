
package com.example.android.renderscript_neuralnet;

import android.content.Context;
import android.content.res.AssetManager;
import android.support.v8.renderscript.Allocation;
import android.support.v8.renderscript.Element;
import android.support.v8.renderscript.RenderScript;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;


public class BatchNormalization extends NeuralNetLayerBase {
    private int size;
    private float[] gamma;
    private float[] beta;
    private float[] avg_mean;
    private float[] avg_var;

    // RS kernel to conduct batch normalization.
    private ScriptC_batchnormalization rs_BN;
    // RS Allocations of the corresponding parameters.
    private Allocation gamma_alloc, beta_alloc, avg_mean_alloc, avg_var_alloc;

    public BatchNormalization(Context ctx, RenderScript rs, int size) {
        super(ctx, rs);

        this.size = size;
        gamma = new float[size];
        beta = new float[size];
        avg_mean = new float[size];
        avg_var = new float[size];

        // Create RS Allocations.
        gamma_alloc = Allocation.createSized(mRS, Element.F32(mRS), size);
        beta_alloc = Allocation.createSized(mRS, Element.F32(mRS), size);
        avg_mean_alloc = Allocation.createSized(mRS, Element.F32(mRS), size);
        avg_var_alloc = Allocation.createSized(mRS, Element.F32(mRS), size);

        // Initialize the BatchNormalization kernel;
        rs_BN = new ScriptC_batchnormalization(mRS);

        // Set the global variables for the RS kernel.
        rs_BN.set_beta_alloc(beta_alloc);
        rs_BN.set_gamma_alloc(gamma_alloc);
        rs_BN.set_mean_alloc(avg_mean_alloc);
        rs_BN.set_var_alloc(avg_var_alloc);
        rs_BN.set_size(size);
    }

    // Load the data from file and transfer to corresponding Allocations.
    public void loadModel(String path) throws IOException {
        mInputStream = mContext.getAssets().open(path + "/gamma", AssetManager.ACCESS_BUFFER);
        ByteBuffer bb = readInput(mInputStream);
        FloatBuffer.wrap(gamma).put(bb.asFloatBuffer());
        gamma_alloc.copyFrom(gamma);

        mInputStream = mContext.getAssets().open(path + "/beta", AssetManager.ACCESS_BUFFER);
        bb = readInput(mInputStream);
        FloatBuffer.wrap(beta).put(bb.asFloatBuffer());
        beta_alloc.copyFrom(beta);

        mInputStream = mContext.getAssets().open(path + "/avg_mean", AssetManager.ACCESS_BUFFER);
        bb = readInput(mInputStream);
        FloatBuffer.wrap(avg_mean).put(bb.asFloatBuffer());
        avg_mean_alloc.copyFrom(avg_mean);

        mInputStream = mContext.getAssets().open(path + "/avg_var", AssetManager.ACCESS_BUFFER);
        bb = readInput(mInputStream);
        FloatBuffer.wrap(avg_var).put(bb.asFloatBuffer());
        avg_var_alloc.copyFrom(avg_var);

        mInputStream.close();
        Log.v(TAG, "BatchNormalization loaded: " + gamma[0] + " " + beta[0] + " " + avg_var[0] + " " + avg_mean[0]);
    }


    public void process(Allocation input) {
        float[] data, data2;
        long time = System.currentTimeMillis();
        // Execute the BatchNormalization kernel.
        rs_BN.forEach_process(input, input);

        // Log time;
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            normalizeTime += time;
            Log.v(TAG, "BatchNormalization, size: " + size + " process time: " + time);
        }
    }
}

