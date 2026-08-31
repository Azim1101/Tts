package com.swara.app.dsp

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin windowed-sinc rational resampler (polyphase interpolation).
 *
 * This mirrors `scipy.signal.resample_poly` behaviour closely enough for the
 * reference pre-processing step (target ~0.1% relative error):
 *   - low-pass cutoff = min(input Nyquist, output Nyquist)
 *   - Kaiser window (beta≈12)
 *   - each output sample is normalised by the sum of the taps actually used,
 *     which removes DC gain ripple near the edges.
 *
 * @property tapsPerSide half-width of the filter kernel, in input samples.
 *   ~24 taps total gives the compact kernel suggested by the spec; bumping it
 *   improves stop-band attenuation at the cost of a little CPU.
 */
object WindowedSincResampler {

    const val KAISER_BETA = 12.0
    private const val TAPS_PER_SIDE = 24

    fun resample(input: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (srcRate == dstRate) return input.copyOf()
        if (input.isEmpty()) return FloatArray(0)

        val ratio = dstRate.toDouble() / srcRate.toDouble()     // output / input
        // Anti-aliasing bandwidth (cycles per input sample):
        val bw = 0.5 * min(1.0, ratio)
        // Prevent an (unlikely) degenerate cutoff.
        val cutoff = bw.coerceIn(1e-9, 0.5)

        val window = kaiser(2 * TAPS_PER_SIDE + 1, KAISER_BETA)

        val outLen = (input.size * dstRate.toLong() / srcRate.toLong()).toInt()
        val out = FloatArray(outLen)

        for (n in 0 until outLen) {
            val pos = n * srcRate.toDouble() / dstRate.toDouble()
            val k0 = floor(pos - TAPS_PER_SIDE).toInt()
            val k1 = ceil(pos + TAPS_PER_SIDE).toInt()

            var acc = 0.0
            var wsum = 0.0
            for (k in k0..k1) {
                if (k < 0 || k >= input.size) continue
                val t = pos - k
                val widx = (t + TAPS_PER_SIDE).roundToInt()
                if (widx < 0 || widx >= window.size) continue
                val weight = sinc(2.0 * cutoff * t) * window[widx]
                acc += input[k].toDouble() * weight
                wsum += weight
            }
            out[n] = if (wsum != 0.0) (acc / wsum).toFloat() else 0.0f
        }
        return out
    }

    private fun sinc(x: Double): Double = if (abs(x) < 1e-12) 1.0 else sin(x * PI) / (x * PI)

    /** Length-[N] Kaiser window, normalised to peak 1.0, beta = b. */
    private fun kaiser(n: Int, b: Double): DoubleArray {
        val w = DoubleArray(n)
        val denom = besselI0(b)
        if (n == 1) {
            w[0] = 1.0
            return w
        }
        for (i in 0 until n) {
            val x = 2.0 * i / (n - 1) - 1.0
            w[i] = besselI0(b * sqrt(1.0 - x * x)) / denom
        }
        return w
    }

    /** Modified Bessel function of the first kind, order 0, via the power series (stable). */
    private fun besselI0(x: Double): Double {
        val ax = abs(x)
        if (ax < 1e-9) return 1.0
        var term = 1.0
        var sum = 1.0
        var k = 1
        val half = ax / 2.0
        while (k < 64) {
            term *= (half / k) * (half / k)
            sum += term
            if (term < 1e-15 * sum) break
            k++
        }
        return sum
    }
}
