package com.tmm.mnisttensorflow

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Arrays
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DigitClassifier(private val context: Context) {
  private var interpreter: Interpreter? = null
  var isInitialized = false
    private set

  /** Executor to run inference task in the background */
  private val executorService: ExecutorService = Executors.newCachedThreadPool()

  private var inputImageWidth: Int = 0 // will be inferred from TF Lite model
  private var inputImageHeight: Int = 0 // will be inferred from TF Lite model
  private var modelInputSize: Int = 0 // will be inferred from TF Lite model

  fun initialize(): Task<Void?> {
    val task = TaskCompletionSource<Void?>()
    executorService.execute {
      try {
        initializeInterpreter()
        task.setResult(null)
      } catch (e: IOException) {
        task.setException(e)
      }
    }
    return task.task
  }

  @Throws(IOException::class)
  private fun initializeInterpreter() {
    // Load the TF Lite model
    val assetManager = context.assets
    val model = loadModelFile(assetManager)

    // Initialize TF Lite Interpreter with NNAPI enabled
    val options = Interpreter.Options()
    // options.setUseNNAPI(true)
    val interpreter = Interpreter(model, options)

    // Read input shape from model file
    val inputShape = interpreter.getInputTensor(0).shape()
    inputImageWidth = inputShape[1]
    inputImageHeight = inputShape[2]
    modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * PIXEL_SIZE

    // Finish interpreter initialization
    this.interpreter = interpreter
    isInitialized = true
    Log.d(TAG, "Initialized TFLite interpreter.")
  }

  @Throws(IOException::class)
  private fun loadModelFile(assetManager: AssetManager): ByteBuffer {
    val fileDescriptor = assetManager.openFd(MODEL_FILE)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
  }

  private fun classify(bitmap: Bitmap): String {
    if (!isInitialized) {
      throw IllegalStateException("TF Lite Interpreter is not initialized yet.")
    }

    var startTime: Long
    var elapsedTime: Long

    // Preprocessing: resize the input
    startTime = System.nanoTime()

    var resizedImage = getScaledBitmap(bitmap)
    // resizedImage = fastblur(resizedImage, 1f, 1)
    val pixels = convertBitmapToPixels(resizedImage)
    val byteBuffer = convertPixelsToByteBuffer(pixels)

    Log.d("YYYY", Arrays.toString(pixels))
    elapsedTime = (System.nanoTime() - startTime) / 1000000
    Log.d(TAG, "Preprocessing time = " + elapsedTime + "ms")

    startTime = System.nanoTime()
    val result = Array(1) { FloatArray(OUTPUT_CLASSES_COUNT) }
    interpreter?.run(byteBuffer, result)
    elapsedTime = (System.nanoTime() - startTime) / 1000000
    Log.d(TAG, "Inference time = " + elapsedTime + "ms")

    return getOutputString(result[0])
  }

  fun getScaledBitmap(bitmap: Bitmap): Bitmap {
    return Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, true)
  }
  fun classifyAsync(bitmap: Bitmap): Task<String> {
    val task = TaskCompletionSource<String>()
    executorService.execute {
      val result = classify(bitmap)
      task.setResult(result)
    }
    return task.task
  }

  fun close() {
    executorService.execute {
      interpreter?.close()
      Log.d(TAG, "Closed TFLite interpreter.")
    }
  }

  private fun convertPixelsToByteBuffer(pixels: FloatArray): ByteBuffer {
    val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
    byteBuffer.order(ByteOrder.nativeOrder())
    for (pixelValue in pixels) {
      byteBuffer.putFloat(pixelValue)
    }
    return byteBuffer
  }

  fun convertBitmapToPixels(bitmap: Bitmap): FloatArray {
    val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
    byteBuffer.order(ByteOrder.nativeOrder())

    val pixels = IntArray(inputImageWidth * inputImageHeight)
    bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    val retPixels = FloatArray(pixels.size)

    pixels.forEachIndexed { index, pixelValue ->
      val r = (pixelValue shr 16 and 0xFF)
      val g = (pixelValue shr 8 and 0xFF)
      val b = (pixelValue and 0xFF)

      // Convert RGB to grayscale and normalize pixel value to [0..1]
      val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
      retPixels[index] = normalizedPixelValue
    }

    return retPixels
    // return byteBuffer
  }

  private fun getOutputString(output: FloatArray): String {
    val maxIndex = output.indices.maxByOrNull { output[it] } ?: -1
    return "Prediction Result: %d\nConfidence: %2f".format(maxIndex, output[maxIndex])
  }

  fun fastblur(sentBitmap: Bitmap, scale: Float, radius: Int): Bitmap {
    var sentBitmap = sentBitmap
    val width = Math.round(sentBitmap.width * scale)
    val height = Math.round(sentBitmap.height * scale)
    sentBitmap = Bitmap.createScaledBitmap(sentBitmap, width, height, false)
    val bitmap = sentBitmap.copy(sentBitmap.config, true)

    val w = bitmap.width
    val h = bitmap.height
    val pix = IntArray(w * h)
    Log.e("pix", w.toString() + " " + h + " " + pix.size)
    bitmap.getPixels(pix, 0, w, 0, 0, w, h)
    val wm = w - 1
    val hm = h - 1
    val wh = w * h
    val div = radius + radius + 1
    val r = IntArray(wh)
    val g = IntArray(wh)
    val b = IntArray(wh)
    var rsum: Int
    var gsum: Int
    var bsum: Int
    var x: Int
    var y: Int
    var i: Int
    var p: Int
    var yp: Int
    var yi: Int
    var yw: Int
    val vmin = IntArray(Math.max(w, h))
    var divsum = div + 1 shr 1
    divsum *= divsum
    val dv = IntArray(256 * divsum)
    i = 0
    while (i < 256 * divsum) {
      dv[i] = i / divsum
      i++
    }
    yi = 0
    yw = yi
    val stack = Array(div) { IntArray(3) }
    var stackpointer: Int
    var stackstart: Int
    var sir: IntArray
    var rbs: Int
    val r1 = radius + 1
    var routsum: Int
    var goutsum: Int
    var boutsum: Int
    var rinsum: Int
    var ginsum: Int
    var binsum: Int
    y = 0
    while (y < h) {
      bsum = 0
      gsum = bsum
      rsum = gsum
      boutsum = rsum
      goutsum = boutsum
      routsum = goutsum
      binsum = routsum
      ginsum = binsum
      rinsum = ginsum
      i = -radius
      while (i <= radius) {
        p = pix[yi + Math.min(wm, Math.max(i, 0))]
        sir = stack[i + radius]
        sir[0] = p and 0xff0000 shr 16
        sir[1] = p and 0x00ff00 shr 8
        sir[2] = p and 0x0000ff
        rbs = r1 - Math.abs(i)
        rsum += sir[0] * rbs
        gsum += sir[1] * rbs
        bsum += sir[2] * rbs
        if (i > 0) {
          rinsum += sir[0]
          ginsum += sir[1]
          binsum += sir[2]
        } else {
          routsum += sir[0]
          goutsum += sir[1]
          boutsum += sir[2]
        }
        i++
      }
      stackpointer = radius
      x = 0
      while (x < w) {
        r[yi] = dv[rsum]
        g[yi] = dv[gsum]
        b[yi] = dv[bsum]
        rsum -= routsum
        gsum -= goutsum
        bsum -= boutsum
        stackstart = stackpointer - radius + div
        sir = stack[stackstart % div]
        routsum -= sir[0]
        goutsum -= sir[1]
        boutsum -= sir[2]
        if (y == 0) {
          vmin[x] = Math.min(x + radius + 1, wm)
        }
        p = pix[yw + vmin[x]]
        sir[0] = p and 0xff0000 shr 16
        sir[1] = p and 0x00ff00 shr 8
        sir[2] = p and 0x0000ff
        rinsum += sir[0]
        ginsum += sir[1]
        binsum += sir[2]
        rsum += rinsum
        gsum += ginsum
        bsum += binsum
        stackpointer = (stackpointer + 1) % div
        sir = stack[stackpointer % div]
        routsum += sir[0]
        goutsum += sir[1]
        boutsum += sir[2]
        rinsum -= sir[0]
        ginsum -= sir[1]
        binsum -= sir[2]
        yi++
        x++
      }
      yw += w
      y++
    }
    x = 0
    while (x < w) {
      bsum = 0
      gsum = bsum
      rsum = gsum
      boutsum = rsum
      goutsum = boutsum
      routsum = goutsum
      binsum = routsum
      ginsum = binsum
      rinsum = ginsum
      yp = -radius * w
      i = -radius
      while (i <= radius) {
        yi = Math.max(0, yp) + x
        sir = stack[i + radius]
        sir[0] = r[yi]
        sir[1] = g[yi]
        sir[2] = b[yi]
        rbs = r1 - Math.abs(i)
        rsum += r[yi] * rbs
        gsum += g[yi] * rbs
        bsum += b[yi] * rbs
        if (i > 0) {
          rinsum += sir[0]
          ginsum += sir[1]
          binsum += sir[2]
        } else {
          routsum += sir[0]
          goutsum += sir[1]
          boutsum += sir[2]
        }
        if (i < hm) {
          yp += w
        }
        i++
      }
      yi = x
      stackpointer = radius
      y = 0
      while (y < h) {
        // Preserve alpha channel: ( 0xff000000 & pix[yi] )
        pix[yi] = -0x1000000 and pix[yi] or (dv[rsum] shl 16) or (dv[gsum] shl 8) or dv[bsum]
        rsum -= routsum
        gsum -= goutsum
        bsum -= boutsum
        stackstart = stackpointer - radius + div
        sir = stack[stackstart % div]
        routsum -= sir[0]
        goutsum -= sir[1]
        boutsum -= sir[2]
        if (x == 0) {
          vmin[y] = Math.min(y + r1, hm) * w
        }
        p = x + vmin[y]
        sir[0] = r[p]
        sir[1] = g[p]
        sir[2] = b[p]
        rinsum += sir[0]
        ginsum += sir[1]
        binsum += sir[2]
        rsum += rinsum
        gsum += ginsum
        bsum += binsum
        stackpointer = (stackpointer + 1) % div
        sir = stack[stackpointer]
        routsum += sir[0]
        goutsum += sir[1]
        boutsum += sir[2]
        rinsum -= sir[0]
        ginsum -= sir[1]
        binsum -= sir[2]
        yi += w
        y++
      }
      x++
    }
    Log.e("pix", w.toString() + " " + h + " " + pix.size)
    bitmap.setPixels(pix, 0, w, 0, 0, w, h)
    return bitmap
  }

  companion object {
    private const val TAG = "DigitClassifier"

    private const val MODEL_FILE = "mnist.tflite"

    private const val FLOAT_TYPE_SIZE = 4
    private const val PIXEL_SIZE = 1

    private const val OUTPUT_CLASSES_COUNT = 10
  }
}
