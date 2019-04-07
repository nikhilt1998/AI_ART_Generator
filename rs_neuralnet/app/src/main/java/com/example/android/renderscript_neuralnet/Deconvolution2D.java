
package com.example.android.renderscript_neuralnet;

import android.content.Context;
import android.content.res.AssetManager;
import android.support.v8.renderscript.Allocation;
import android.support.v8.renderscript.Element;
import android.support.v8.renderscript.RenderScript;
import android.support.v8.renderscript.ScriptIntrinsicBLAS;
import android.support.v8.renderscript.Type;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;


public class Deconvolution2D extends NeuralNetLayerBase {
    
    public int outH, outW;

    private int in_channels, out_channels;
    private int ksize, stride, pad;
    private float[] W;
    private float[] b;

    // The padded dimension to satisfy alignment requirement for certain GPUs.
    private int padded_Y_blas;

    private ScriptC_deconvolve2d mConvovle;
    private Allocation W_alloc, b_alloc;

    public Deconvolution2D(Context ctx, RenderScript rs, int in_channels, int out_channels, int ksize, int stride, int pad) {
        super(ctx, rs);

        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.ksize = ksize;
        this.stride = stride;
        this.pad = pad;
       
        this.W = new float[out_channels * ksize * ksize * in_channels];
        this.b = new float[out_channels];

     
        padded_Y_blas = out_channels * ksize * ksize;
        if (padded_Y_blas % 8 > 0) {
            padded_Y_blas = (padded_Y_blas / 8 + 1) * 8;
        }

        // Create Allocations for W and b.
        W_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), in_channels, padded_Y_blas));
        b_alloc = Allocation.createSized(mRS, Element.F32(mRS), out_channels);

        // Initialize the 2D deconvolution kernel;
        mConvovle = new ScriptC_deconvolve2d(mRS);

        // Set the global variables for the RS kernel.
        mConvovle.set_kernel_h(ksize);
        mConvovle.set_kernel_w(ksize);
        mConvovle.set_step_x(stride);
        mConvovle.set_step_y(stride);
        mConvovle.set_pad_h(pad);
        mConvovle.set_pad_w(pad);

        mConvovle.set_beta_alloc(b_alloc);

    }

    // Load the data from file and transfer to corresponding Allocations.
    public void loadModel(String path) throws IOException {
        mInputStream = mContext.getAssets().open(path + "/W", AssetManager.ACCESS_BUFFER);
        ByteBuffer bb = readInput(mInputStream);
        FloatBuffer.wrap(W).put(bb.asFloatBuffer());
        // Tranpose W after loading the data.
        float[] w_trans = new float[in_channels * padded_Y_blas];
        for (int i = 0; i < out_channels * ksize * ksize; i++) {
            for (int j = 0; j < in_channels; j++) {
                w_trans[i * in_channels + j] = W[j * out_channels * ksize * ksize + i];
            }
        }
        W_alloc.copyFrom(w_trans);

        mInputStream = mContext.getAssets().open(path + "/b", AssetManager.ACCESS_BUFFER);
        bb = readInput(mInputStream);
        FloatBuffer.wrap(b).put(bb.asFloatBuffer());
        b_alloc.copyFrom(b);

        mInputStream.close();
        Log.v(TAG, "Deconvolution2D loaded: " + b[0]);
    }

 
    public Allocation process(Allocation input, int col_h, int col_w) {
        // Create the output Allocation for SGEMM operation.
        Allocation out_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), input.getType().getX(), W_alloc.getType().getY()));

        long time = System.currentTimeMillis();
        Log.v(TAG, "Deconvolution2D: " + input.getType().getX() + " " + input.getType().getY() + " " + W_alloc.getType().getX() + " " +  W_alloc.getType().getY());
        // Conduct the deconvolution by matrix multiplication, using SGEMM (BLAS API).
        mBlas.SGEMM(ScriptIntrinsicBLAS.NO_TRANSPOSE, ScriptIntrinsicBLAS.NO_TRANSPOSE,
                1.0f, W_alloc, input, 0.0f, out_alloc);
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            sgemmTime += time;
            Log.v(TAG, "Deconvolution2D, channels: " + in_channels + ", " + out_channels + " size: " + col_h + ", " + col_w + " SGEMM process time: " + time);
        }

        Log.v(TAG, "Deconvolution2D: SGEMM");
        // Calculate the dimensions of image after deconvolution.
        int img_h = ConvolveUtil.get_deconv_outsize(col_h, ksize, stride, pad);
        int img_w = ConvolveUtil.get_deconv_outsize(col_w, ksize, stride, pad);

        // Set the global input variables for the RS kernel.
        mConvovle.set_col_h(col_h);
        mConvovle.set_col_w(col_w);
        mConvovle.set_col_channel(out_channels);
        mConvovle.set_img_channel(out_channels);
        mConvovle.set_img_h(img_h);
        mConvovle.set_img_w(img_w);
        mConvovle.set_col_alloc(out_alloc);

        // Calculate the dimensions of the padded image.
        int padded_h = img_h + 2 * pad;
        int padded_w = img_w + 2 * pad;
        // Create Allocation to hold the padded image.
        Allocation img_padded = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), padded_h * padded_w, out_channels));
        // Initialize the padded Allocation to zero.
        mConvovle.forEach_zero(img_padded, img_padded);
        mConvovle.set_padded_alloc(img_padded);

        // Create output image Allocation.
        Allocation img_alloc = Allocation.createTyped(mRS,
                Type.createXY(mRS, Element.F32(mRS), img_h * img_w, out_channels));
        mConvovle.set_img_alloc(img_alloc);
        time = System.currentTimeMillis();

        // Invoke col2im kernel, to transform column image to padded image:
        mConvovle.invoke_col2im();
       

        // Invoked the unpadding kernel.
        mConvovle.invoke_unpadd();
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            col2imTime += time;
            Log.v(TAG, "Deconvolution2D, channels: " + in_channels + ", " + out_channels + " size: " + col_h + ", " + col_w + " col2im process time: " + time);
        }

        time = System.currentTimeMillis();
        // Add beta to the results for each channel.
        mConvovle.forEach_addBeta(img_alloc, img_alloc);
        if (LOG_TIME) {
            mRS.finish();
            time = System.currentTimeMillis() - time;
            betaTime += time;
            Log.v(TAG, "Deconvolution2D, channels: " + in_channels + ", " + out_channels + " size: " + col_h + ", " + col_w + " addBeta process time: " + time);
        }

        // Destroy the intermediate Allocations.
        out_alloc.destroy();
        img_padded.destroy();

        // Update the output dimensions.
        outH = img_h;
        outW = img_w;

        return img_alloc;
    }
}
