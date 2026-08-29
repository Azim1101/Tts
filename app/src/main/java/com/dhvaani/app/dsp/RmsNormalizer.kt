package com.dhvaani.app.dsp

import kotlin.math.sqrt

/** Result of the reference amplitude normalisation step. */
data class RmsResult(
    /** Normalised audio (24 kHz). */
    val signal: FloatArray,
    /** Root-mean-square of the ORIGINAL (pre-boost) audio. */
    val originalRms: Float,
    /** True when the signal was amplified up to the target RMS. */
    val boosted: Boolean
)

/**
 * Reference pre-processing RMS normalisation.
 *
 * If the signal RMS is below [DspConstants.TARGET_RMS] it is scaled UP so that
 * its RMS equals the target (this also amplifies any background noise — see the
 * honest-limitations list). Otherwise it is left untouched. The vocoder uses
 * [originalRms] to undo this boost before the final output clip.
 */
object RmsNormalizer {

    fun apply(input: FloatArray): RmsResult {
        if (input.isEmpty()) return RmsResult(input, 0f, false)
        val n = input.size
        var sum = 0.0
        for (v in input) sum += v.toDouble() * v.toDouble()
        val rms = sqrt(sum / n).toFloat()
        if (rms < DspConstants.TARGET_RMS && rms > 1e-9f) {
            val scale = DspConstants.TARGET_RMS / rms
            val boosted = FloatArray(n)
            for (i in 0 until n) boosted[i] = input[i] * scale
            return RmsResult(boosted, rms, true)
        }
        return RmsResult(input.copyOf(), rms, false)
    }

    fun rms(input: FloatArray): Float {
        if (input.isEmpty()) return 0f
        var sum = 0.0
        for (v in input) sum += v.toDouble() * v.toDouble()
        return sqrt(sum / input.size).toFloat()
    }

    /** Multiply every sample by [factor] (used to undo an RMS boost). */
    fun scale(input: FloatArray, factor: Float): FloatArray {
        val out = FloatArray(input.size)
        for (i in input.indices) out[i] = input[i] * factor
        return out
    }
}
