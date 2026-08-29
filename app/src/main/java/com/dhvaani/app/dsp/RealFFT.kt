package com.dhvaani.app.dsp

import kotlin.math.cos
import kotlin.math.sin

/**
 * Pure-Kotlin in-place FFT (radix-2, iterative DIT) plus thin helpers to compute
 * real-valued forward/backward transforms used by the Vocos frontend and the
 * ISTFT vocoder. No external math/graphics library is used.
 *
 * The transforms operate on Double for extra precision; results are cast to
 * Float only where the reference pipeline uses 32-bit tensors.
 */
object RealFFT {

    /**
     * In-place iterative Cooley-Tukey FFT.
     * @param re real parts, length n (power of two)
     * @param im imaginary parts, length n
     * @param inverse if true performs the inverse transform (no 1/n scaling here;
     *                callers scale by 1/n afterwards)
     */
    fun fft(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
        val n = re.size
        require(n and (n - 1) == 0) { "FFT length must be a power of two, got $n" }

        // Bit-reversal permutation.
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
            var m = n shr 1
            while (j and m != 0) { j = j xor m; m = m shr 1 }
            j = j xor m
        }

        // Danielson-Lanczos butterflies.
        var len = 2
        while (len <= n) {
            val ang = (if (inverse) 2.0 else -2.0) * Math.PI / len
            val wRe = cos(ang)
            val wIm = sin(ang)
            var i = 0
            while (i < n) {
                var curRe = 1.0
                var curIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
                    val vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
                    re[i + k] = uRe + vRe
                    im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe
                    im[i + k + len / 2] = uIm - vIm
                    val nextRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nextRe
                }
                i += len
            }
            len = len shl 1
        }

        if (inverse) {
            for (i in 0 until n) {
                re[i] /= n
                im[i] /= n
            }
        }
    }

    /**
     * Real forward transform: returns packed [N_FREQ] magnitude array. Only the
     * forward transform is needed for the mel frontend. The complex spectrum's
     * phase is not required there.
     */
    fun rfftMag(input: FloatArray, n: Int = DspConstants.N_FFT): FloatArray {
        val re = DoubleArray(n)
        for (i in 0 until n) re[i] = input[i].toDouble()
        val im = DoubleArray(n)
        fft(re, im, false)
        val out = FloatArray(n / 2 + 1)
        for (i in out.indices) {
            out[i] = kotlin.math.hypot(re[i], im[i]).toFloat()
        }
        return out
    }

    /**
     * Real inverse transform from a full complex spectrum (length n). Takes the
     * (already inverse-scaled) spectra and reconstructs the real time-domain
     * frame of length n.
     */
    fun irfft(re: DoubleArray, im: DoubleArray, n: Int): FloatArray {
        fft(re, im, true)
        val out = FloatArray(n)
        for (i in 0 until n) out[i] = re[i].toFloat()
        return out
    }
}
