package com.dhvaani.app.dsp

import com.dhvaani.app.onnx.OnnxEngine
import com.dhvaani.app.onnx.VocosHead
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin

/**
 * Vocos vocoder in pure Kotlin. The ConvNeXt backbone runs through onnxruntime;
 * the linear head, magnitude/phase split and overlap-add inverse STFT are done
 * on the CPU in Java/Kotlin (no numpy).
 *
 * Input `mels` is [N_MELS x numFrames] laid out as mel-channel-major
 * (`mels[mel*numFrames + frame]`), matching the ONNX `[1,100,T]` expectation.
 */
class VocosVocoder(
    private val engine: OnnxEngine,
    private val head: VocosHead
) {

    private val nFft = head.nFft
    private val hop = head.hop
    private val nFreq = nFft / 2 + 1
    private val outDim = head.linearBias.size          // 1026
    private val win = head.window                       // periodic Hann, length win_length
    private val halfN = (nFft - hop) / 2                // 'same' centre padding per side

    /**
     * @param mels      mel-channel-major log-mel, length = 100 * numFrames
     * @param numFrames number of mel frames (== output audio frames)
     * @return PCM at 24 kHz
     */
    fun synthesize(mels: FloatArray, numFrames: Int): FloatArray {
        require(mels.size == DspConstants.N_MELS * numFrames) { "mels size mismatch" }

        // 1) Backbone -> (numFrames, 512)
        val hidden = engine.vocoderBackbone(mels)
        require(hidden.size == numFrames * DspConstants.HIDDEN_DIM) { "hidden size mismatch" }

        // 2) Linear head -> (numFrames, 1026)
        val headOut = FloatArray(numFrames * outDim)
        val weight = head.linearWeight
        val bias = head.linearBias
        for (frame in 0 until numFrames) {
            val hidBase = frame * DspConstants.HIDDEN_DIM
            val outBase = frame * outDim
            for (o in 0 until outDim) {
                val wBase = o * DspConstants.HIDDEN_DIM
                var acc = 0.0
                for (h in 0 until DspConstants.HIDDEN_DIM) {
                    acc += hidden[hidBase + h].toDouble() * weight[wBase + h].toDouble()
                }
                headOut[outBase + o] = (acc + bias[o]).toFloat()
            }
        }

        // 3) magnitude = exp(clip(head[:513], -inf, log(100))), phase = head[513:]
        val mag = FloatArray(numFrames * nFreq)
        val phase = FloatArray(numFrames * nFreq)
        val log100 = ln(100.0).toFloat()
        for (i in 0 until numFrames) {
            val base = i * outDim
            for (f in 0 until nFreq) {
                val m = headOut[base + f]
                mag[i * nFreq + f] = exp(if (m > log100) log100 else m).toFloat()
                phase[i * nFreq + f] = headOut[base + nFreq + f]
            }
        }

        // 4) Overlap-add inverse STFT (pad='same')
        return istftSame(mag, phase, numFrames)
    }

    private fun istftSame(mag: FloatArray, phase: FloatArray, numFrames: Int): FloatArray {
        val total = (numFrames - 1) * hop + nFft     // OLA length before trimming
        val outLen = total - 2 * halfN                // 'same' -> numFrames * hop
        val out = FloatArray(outLen)
        val wsum = FloatArray(outLen)

        val re = DoubleArray(nFft)
        val im = DoubleArray(nFft)

        for (frame in 0 until numFrames) {
            val mBase = frame * nFreq
            val half = nFft / 2
            // Build the Hermitian-symmetric complex spectrum from mag + phase.
            // Bins 0..(nFreq-1) are the DFT bins 0..half (DC .. Nyquist).
            for (f in 0 until nFreq) {
                val m = mag[mBase + f]
                val p = phase[mBase + f]
                re[f] = m * cos(p.toDouble())
                im[f] = m * sin(p.toDouble())
            }
            // Hermitian mirror for the upper half: S[n-f] = conj(S[f]).
            for (f in 1 until half) {
                re[nFft - f] = re[f]
                im[nFft - f] = -im[f]
            }
            // DC and Nyquist bins are real.
            im[0] = 0.0
            im[half] = 0.0

            val frameTd = RealFFT.irfft(re, im, nFft)

            val start = frame * hop - halfN
            for (k in 0 until nFft) {
                val idx = start + k
                if (idx < 0 || idx >= outLen) continue
                val w = win[k]
                out[idx] = out[idx] + frameTd[k] * w
                wsum[idx] = wsum[idx] + w * w
            }
        }

        // Normalise by the window^2 envelope (clamp to avoid div-by-0).
        for (i in 0 until outLen) {
            val env = wsum[i]
            out[i] = if (env > 1e-11f) out[i] / env else 0.0f
        }
        return out
    }
}
