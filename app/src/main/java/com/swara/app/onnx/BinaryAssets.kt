package com.swara.app.onnx

import com.swara.app.dsp.DspConstants
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Little-endian raw readers for the small feature `.bin` assets. These are
 * produced by `scripts/make_assets.py` from the original `.npz` files.
 *
 * File layouts (all multi-byte values little-endian):
 *
 *   mel_fb.bin
 *     int32[3]   n_fft, hop, n_mels
 *     float32[513*100]   mel filterbank fb, row-major (freq x mel)
 *     float32[1024]      analysis window (periodic Hann)
 *
 *   vocos_head.bin
 *     int32[3]   n_fft, hop, win_length
 *     float32[1026*512]  linear_weight, row-major (out x in)
 *     float32[1026]      linear_bias
 *     float32[1024]      synthesis window
 */
class MelFilterbank(
    val nFft: Int,
    val hop: Int,
    val nMels: Int,
    val fb: FloatArray,       // 513 * nMels
    val window: FloatArray    // nFft
) {
    companion object {
        fun fromBytes(input: InputStream): MelFilterbank {
            val data = ByteBuffer.wrap(input.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val nFft = data.int
            val hop = data.int
            val nMels = data.int
            val freq = nFft / 2 + 1
            val fb = FloatArray(freq * nMels)
            val window = FloatArray(nFft)
            data.asFloatBuffer().get(fb)
            // advance past fb
            data.position(data.position() + fb.size * 4)
            data.asFloatBuffer().get(window)
            return MelFilterbank(nFft, hop, nMels, fb, window)
        }
    }
}

class VocosHead(
    val nFft: Int,
    val hop: Int,
    val winLength: Int,
    val linearWeight: FloatArray,  // 1026 * 512, row-major out x in
    val linearBias: FloatArray,    // 1026
    val window: FloatArray         // 1024
) {
    companion object {
        val OUT_DIM = DspConstants.OUT_DIM
        val HIDDEN_DIM = DspConstants.HIDDEN_DIM

        fun fromBytes(input: InputStream): VocosHead {
            val data = ByteBuffer.wrap(input.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val nFft = data.int
            val hop = data.int
            val winLength = data.int
            val weight = FloatArray(OUT_DIM * HIDDEN_DIM)
            val bias = FloatArray(OUT_DIM)
            val window = FloatArray(winLength)
            data.asFloatBuffer().get(weight)
            data.position(data.position() + weight.size * 4)
            data.asFloatBuffer().get(bias)
            data.position(data.position() + bias.size * 4)
            data.asFloatBuffer().get(window)
            return VocosHead(nFft, hop, winLength, weight, bias, window)
        }
    }
}
