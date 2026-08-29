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
 *
 * ## Output headroom (v0.7.5)
 *
 * Previously the vocoder-output-then-unboost path could push peaks to ±1.0
 * which hard-clips on playback (visible in v0.5 benchmark: outputs had
 * `peak 1.0` even though reference was `peak 0.51`). We now keep a small
 * headroom (PEAK_HEADROOM = 0.95) so even after a vocoder overshoot the output
 * does not clip. The downside is ~0.5 dB of level — inaudible on phone
 * speakers.
 */
object RmsNormalizer {

    /** Output peak will be at most ±PEAK_HEADROOM after unboost. */
    const val PEAK_HEADROOM = 0.95f

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

    /**
     * Multiply every sample by [factor] and clamp into
     * [-PEAK_HEADROOM, +PEAK_HEADROOM] so the final PCM stream never hits
     * the rails. `PEAK_HEADROOM = 0.95` is the safety margin the
     * vocoder-induced overshoot needs.
     */
    fun scale(input: FloatArray, factor: Float): FloatArray {
        val out = FloatArray(input.size)
        val lo = -PEAK_HEADROOM
        val hi = PEAK_HEADROOM
        for (i in input.indices) {
            val v = input[i] * factor
            out[i] = if (v > hi) hi else if (v < lo) lo else v
        }
        return out
    }
}
