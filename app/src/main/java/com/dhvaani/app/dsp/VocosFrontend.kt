package com.dhvaani.app.dsp

import kotlin.math.ln
import kotlin.math.roundToInt

/**
 * Vocos log-mel frontend (pure Kotlin, no numpy/torch).
 *
 * Given 24 kHz mono audio it produces the [frames x 100] natural-log mel
 * spectrogram used for speaker conditioning:
 *   1. zero-pad with N_FFT/2 on both ends
 *   2. frame with hop 256, size 1024, periodic-Hann window (from mel_fb.bin)
 *   3. per frame: rfft(frame * window) -> magnitude (513 bins)
 *   4. mel = mag @ mel_filterbank (513 x 100)
 *   5. logmel = ln(clip(mel, 1e-7, +inf))
 *   6. trim/pad to the canonical frame count, edge-replicating the last frame
 *
 * The caller multiplies by FEAT_SCALE to obtain `prompt_features`.
 */
class VocosFrontend(
    private val melFilterbank: FloatArray,   // 513 * 100, row-major
    private val window: FloatArray,          // N_FFT length periodic-Hann
    private val nFft: Int = DspConstants.N_FFT,
    private val hop: Int = DspConstants.HOP,
    private val nMels: Int = DspConstants.N_MELS
) {

    val nFreq: Int = nFft / 2 + 1
    private val pad = nFft / 2

    /**
     * @return flattened log-mel, length = frames * nMels. Frames follow the
     *         canonical count derived from the audio duration:
     *         floor((round(dur*SR) + HOP//2) // HOP).
     */
    fun computeLogMel(signal: FloatArray): FloatArray {
        if (signal.isEmpty()) return FloatArray(0)
        val dur = signal.size.toDouble() / DspConstants.SR

        // A zero-padded frame buffer. We index into `padded` and frames slide by `hop`.
        val padded = FloatArray(signal.size + 2 * pad)
        System.arraycopy(signal, 0, padded, pad, signal.size)

        // Raw frame count before the canonical trim.
        val rawFrames = ((padded.size - nFft) / hop) + 1
        val canonicalFrames = canonicalNumFrames(dur)

        val out = FloatArray(canonicalFrames * nMels)
        val magnitude = FloatArray(nFreq)

        val framesToUse: Int
        // We always produce `canonicalFrames`, but if the raw framing produced
        // fewer (very short reference), we simply replicate the last available one.
        framesToUse = canonicalFrames

        var lastFrameData: FloatArray? = null
        for (i in 0 until framesToUse) {
            val frameStart = i * hop
            val data = if (frameStart + nFft <= padded.size) {
                frameMel(padded, frameStart, magnitude)
            } else {
                lastFrameData ?: frameMel(padded, (padded.size - nFft).coerceAtLeast(0), magnitude)
            }
            lastFrameData = data
            System.arraycopy(data, 0, out, i * nMels, nMels)
        }
        return out
    }

    /** floor((round(dur*SR) + HOP//2) // HOP) */
    private fun canonicalNumFrames(dur: Double): Int {
        val samples = (dur * DspConstants.SR).roundToInt()
        val n = samples + hop / 2
        return (n / hop).coerceAtLeast(1)
    }

    /** Computes the log-mel of the frame starting at [start] into [magnitude]. */
    private fun frameMel(padded: FloatArray, start: Int, magnitude: FloatArray): FloatArray {
        // Frame * Hann window.
        val framed = FloatArray(nFft)
        for (k in 0 until nFft) framed[k] = padded[start + k] * window[k]

        val mag = RealFFT.rfftMag(framed, nFft)
        System.arraycopy(mag, 0, magnitude, 0, nFreq)

        val mel = FloatArray(nMels)
        for (m in 0 until nMels) {
            var acc = 0.0
            val base = m * nFreq
            for (f in 0 until nFreq) {
                acc += magnitude[f].toDouble() * melFilterbank[base + f].toDouble()
            }
            val clamped = if (acc < DspConstants.MEL_MIN) DspConstants.MEL_MIN.toDouble() else acc
            mel[m] = ln(clamped).toFloat()
        }
        return mel
    }
}
