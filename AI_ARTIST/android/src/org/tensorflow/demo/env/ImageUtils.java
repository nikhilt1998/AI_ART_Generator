

package org.tensorflow.demo.env;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Environment;
import java.io.File;
import java.io.FileOutputStream;


public class ImageUtils {
  @SuppressWarnings("unused")
  private static final Logger LOGGER = new Logger();
  
  static {
    try {
      System.loadLibrary("tensorflow_demo");
    } catch (UnsatisfiedLinkError e) {
      LOGGER.w("Native library not found, native RGB -> YUV conversion may be unavailable.");
    }
  }

  public static int getYUVByteSize(final int width, final int height) {
    
    final int ySize = width * height;

    final int uvSize = ((width + 1) / 2) * ((height + 1) / 2) * 2;

    return ySize + uvSize;
  }


  public static void saveBitmap(final Bitmap bitmap) {
    saveBitmap(bitmap, "preview.png");
  }

  public static void saveBitmap(final Bitmap bitmap, final String filename) {
    final String root =
        Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "tensorflow";
    LOGGER.i("Saving %dx%d bitmap to %s.", bitmap.getWidth(), bitmap.getHeight(), root);
    final File myDir = new File(root);

    if (!myDir.mkdirs()) {
      LOGGER.i("Make dir failed");
    }

    final String fname = filename;
    final File file = new File(myDir, fname);
    if (file.exists()) {
      file.delete();
    }
    try {
      final FileOutputStream out = new FileOutputStream(file);
      bitmap.compress(Bitmap.CompressFormat.PNG, 99, out);
      out.flush();
      out.close();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
    }
  }

  static final int kMaxChannelValue = 262143;

  // Always prefer the native implementation if available.
  private static boolean useNativeConversion = true;

  public static void convertYUV420ToARGB8888(
      byte[] yData,
      byte[] uData,
      byte[] vData,
      int width,
      int height,
      int yRowStride,
      int uvRowStride,
      int uvPixelStride,
      int[] out) {
    if (useNativeConversion) {
      try {
        convertYUV420ToARGB8888(
            yData, uData, vData, out, width, height, yRowStride, uvRowStride, uvPixelStride, false);
        return;
      } catch (UnsatisfiedLinkError e) {
        LOGGER.w("Native YUV -> RGB implementation not found, falling back to Java implementation");
        useNativeConversion = false;
      }
    }

    int i = 0;
    for (int y = 0; y < height; y++) {
      int pY = yRowStride * y;
      int uv_row_start = uvRowStride * (y >> 1);
      int pUV = uv_row_start;
      int pV = uv_row_start;

      for (int x = 0; x < width; x++) {
        int uv_offset = pUV + (x >> 1) * uvPixelStride;
        out[i++] =
            YUV2RGB(
                convertByteToInt(yData, pY + x),
                convertByteToInt(uData, uv_offset),
                convertByteToInt(vData, uv_offset));
      }
    }
  }

  private static int convertByteToInt(byte[] arr, int pos) {
    return arr[pos] & 0xFF;
  }

  private static int YUV2RGB(int nY, int nU, int nV) {
    nY -= 16;
    nU -= 128;
    nV -= 128;
    if (nY < 0) nY = 0;

  
    final int foo = 1192 * nY;
    int nR = foo + 1634 * nV;
    int nG = foo - 833 * nV - 400 * nU;
    int nB = foo + 2066 * nU;

    nR = Math.min(kMaxChannelValue, Math.max(0, nR));
    nG = Math.min(kMaxChannelValue, Math.max(0, nG));
    nB = Math.min(kMaxChannelValue, Math.max(0, nB));

    return 0xff000000 | ((nR << 6) & 0x00ff0000) | ((nG >> 2) & 0x0000FF00) | ((nB >> 10) & 0xff);
  }

  public static native void convertYUV420SPToARGB8888(
      byte[] input, int[] output, int width, int height, boolean halfSize);

  public static native void convertYUV420ToARGB8888(
      byte[] y,
      byte[] u,
      byte[] v,
      int[] output,
      int width,
      int height,
      int yRowStride,
      int uvRowStride,
      int uvPixelStride,
      boolean halfSize);

  
  public static native void convertYUV420SPToRGB565(
      byte[] input, byte[] output, int width, int height);

  
  public static native void convertARGB8888ToYUV420SP(
      int[] input, byte[] output, int width, int height);

  public static native void convertRGB565ToYUV420SP(
      byte[] input, byte[] output, int width, int height);

  
  public static Matrix getTransformationMatrix(
      final int srcWidth,
      final int srcHeight,
      final int dstWidth,
      final int dstHeight,
      final int applyRotation,
      final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (applyRotation != 0) {
      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

   
    final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

    final int inWidth = transpose ? srcHeight : srcWidth;
    final int inHeight = transpose ? srcWidth : srcHeight;

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) inWidth;
      final float scaleFactorY = dstHeight / (float) inHeight;

      if (maintainAspectRatio) {
        
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        // Scale exactly to fill dst from src.
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    if (applyRotation != 0) {
      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;
  }
}
