
package com.example.android.renderscript_neuralnet;

import android.content.Context;
import android.support.v8.renderscript.RenderScript;
import android.support.v8.renderscript.ScriptIntrinsicBLAS;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public abstract class NeuralNetLayerBase {
    public static final String TAG = "FastStyleModel";
    public static final boolean LOG_TIME = true;

    public InputStream mInputStream;
    public Context mContext;
    public RenderScript mRS;
    public ScriptIntrinsicBLAS mBlas;

    public long sgemmTime = 0;
    public long normalizeTime = 0;
    public long im2colTime = 0;
    public long col2imTime = 0;
    public long betaTime = 0;
    public long conv2dTime = 0;

    public NeuralNetLayerBase(Context ctx, RenderScript rs) {
        mContext = ctx;
        mRS = rs;
        mBlas = ScriptIntrinsicBLAS.create(mRS);
    }

    abstract public void loadModel(String path) throws IOException;

    public ByteBuffer readInput(InputStream inputStream) throws IOException {
        // this dynamically extends to take the bytes you read
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

        // this is storage overwritten on each iteration with bytes
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        // we need to know how may bytes were read to write them to the byteBuffer
        int len = 0;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }

        // and then we can return your byte array.
        return ByteBuffer.wrap(byteBuffer.toByteArray()).order(ByteOrder.nativeOrder());
    }

    public void getBenchmark(BenchmarkResult result) {
        result.sgemmTime += sgemmTime;
        result.normalizeTime += normalizeTime;
        result.im2colTime += im2colTime;
        result.col2imTime += col2imTime;
        result.betaTime += betaTime;
        result.conv2dTime += conv2dTime;

        sgemmTime = 0;
        normalizeTime = 0;
        im2colTime = 0;
        col2imTime = 0;
        betaTime = 0;
        conv2dTime = 0;
    }
}
