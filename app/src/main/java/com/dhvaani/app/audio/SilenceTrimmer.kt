package com.dhvaani.app.audio

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Trim leading/trailing silence from a mono PCM buffer (Float, any sample rate
 * — the algorithms are time-domain and don't care about SR).
 *
 * Used as a pre-processing step on the **reference voice** before it is fed
 * into the DhVaani prompt encoder. Without trimming, a 9 s WAV that contains
 * 1 s of speech + 4 s of silence + 4 s of speech wastes 8 s of prompt budget
 * on noise and degrades the cloned voice (the model tries to "clone" the
 * silence too).
 *
 * ## Algorithm
 *
 * 1. Compute the global peak.
 * 2. The trim threshold is `max(NOISE_FLOOR, 0.04 * peak)` — picks up quiet
 *    speech but rejects electrical noise. Tuned for typical phone recordings.
 * 3. Walk in from the head until the running RMS over a 25 ms window first
 *    crosses the threshold. That is the new `start`.
 * 4. Mirror from the tail.
 * 5. Pad both sides by `HEAD_PAD_S` so the prompt encoder doesn't see a hard
 *    edge (the model uses a Hann window at the boundaries).
 * 6. If the trim would leave < [MIN_DURATION_S] of audio, fall back to the
 *    full signal — better to clone a quiet record than to over-trim.
 *
 * The whole thing is O(n) and allocates two scratch arrays, so it is fine
 * to run on the 24 kHz mono float buffer right after `AudioImporter.decode`
 * or `AudioRecorder.stop()`.
 */
object SilenceTrimmer {

    /** Anything quieter than this (linear, 0..1) is treated as silence. */
    const val NOISE_FLOOR: Float = 0.005f

    /** Window size used for the leading-edge detector. 25 ms. */
    private const val WINDOW_S: Double = 0.025

    /** Pad kept at each side after the trim. 50 ms. */
    private const val HEAD_PAD_S: Double = 0.050

    /** If trim would leave less than this, fall back to the whole signal. */
    const val MIN_DURATION_S: Double = 0.5

    /**
     * @return the input if trimming would have left nothing useful, otherwise
     *         a new [FloatArray] with leading and trailing silence removed.
     */
    fun trim(samples: FloatArray, sampleRate: Int): FloatArray {
        if (samples.isEmpty()) return samples
        if (sampleRate <= 0) return samples

        val n = samples.size
        val window = max(1, (WINDOW_S * sampleRate).toInt())
        val pad = max(0, (HEAD_PAD_S * sampleRate).toInt())

        // Peak
        var peak = 0f
        for (v in samples) {
            val a = abs(v)
            if (a > peak) peak = a
        }
        if (peak < 1e-6f) return samples  // fully silent already — return as-is
        val thr = max(NOISE_FLOOR, 0.04f * peak)

        // First non-silent window from the head.
        var start = 0
        outerHead@ for (i in 0..n - window) {
            var sumSq = 0.0
            for (j in 0 until window) {
                val v = samples[i + j]
                sumSq += v.toDouble() * v.toDouble()
            }
            val rms = sqrt(sumSq / window).toFloat()
            if (rms >= thr) { start = i; break@outerHead }
        }
        // First non-silent window from the tail.
        var end = n
        outerTail@ for (i in n - window downTo 0) {
            var sumSq = 0.0
            for (j in 0 until window) {
                val v = samples[i + j]
                sumSq += v.toDouble() * v.toDouble()
            }
            val rms = sqrt(sumSq / window).toFloat()
            if (rms >= thr) { end = i + window; break@outerTail }
        }
        if (end <= start) return samples

        // Apply head pad
        val lo = max(0, start - pad)
        val hi = min(n, end + pad)
        if (hi - lo < (MIN_DURATION_S * sampleRate).toInt()) {
            // Trim would leave too little audio — keep the whole thing.
            return samples
        }
        if (lo == 0 && hi == n) return samples  // nothing to trim
        return samples.copyOfRange(lo, hi)
    }
}
