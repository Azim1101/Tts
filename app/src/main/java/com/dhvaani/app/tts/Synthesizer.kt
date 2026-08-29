package com.dhvaani.app.tts

import com.dhvaani.app.dsp.DspConstants
import com.dhvaani.app.dsp.RmsNormalizer
import com.dhvaani.app.dsp.RmsResult
import com.dhvaani.app.dsp.VocosFrontend
import com.dhvaani.app.dsp.VocosVocoder
import com.dhvaani.app.dsp.WindowedSincResampler
import com.dhvaani.app.onnx.OnnxEngine
import kotlin.math.ln
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Zero-shot voice cloning pipeline, all on device.
 *
 * Flow (see spec §3):
 *   reference audio -> resample 24k -> RMS normalise -> log-mel (prompt_features)
 *   target text     -> tokenize -> text_encoder -> text_condition
 *   x = randn(T,100) seed 666
 *   speech_condition = [prompt_features; zeros]
 *   Euler flow-matching (t_shift 0.5) using fm_decoder
 *   drop prompt frames -> unscale -> Vocos vocoder -> PCM
 *
 * Long target text is split into sentence chunks, each synthesised separately
 * and concatenated. This keeps the per-call tensor size (and therefore memory
 * and latency) bounded, while the reference conditioning stays constant.
 */
class Synthesizer(
    private val engine: OnnxEngine,
    private val tokenizer: Tokenizer,
    private val frontend: VocosFrontend,
    private val vocoder: VocosVocoder
) {

    /** Reference-derived conditioning, computed once and reused per sentence. */
    private data class PromptInfo(
        val promptFeatures: FloatArray,
        val promptTokenIds: LongArray,
        val promptLen: Int,
        val rmsResult: RmsResult
    )

    fun synthesize(
        reference: FloatArray,
        refSampleRate: Int,
        refTranscript: String,
        targetText: String,
        steps: Int,
        guidance: Float,
        speed: Float,
        onProgress: (done: Int, total: Int) -> Unit
    ): FloatArray {

        val prompt = buildPromptInfo(reference, refSampleRate, refTranscript)

        // Split into sentences and keep only those that yield tokens.
        val chunks = splitSentences(targetText).filter {
            tokenizer.toIds(tokenizer.addPunctuation(it)).isNotEmpty()
        }
        if (chunks.isEmpty()) throw IllegalArgumentException("NO_TOKENS")

        val totalSteps = steps * chunks.size
        var doneSteps = 0
        val outs = ArrayList<FloatArray>(chunks.size)

        for (chunk in chunks) {
            val audio = synthesizeChunk(chunk, prompt, steps, guidance, speed) { done, total ->
                onProgress(doneSteps + done, totalSteps)
            }
            doneSteps += steps
            outs.add(audio)
        }
        return concat(outs)
    }

    // ---------------------------------------------------------------------
    // Reference preprocessing (steps 1-2), computed once.
    // ---------------------------------------------------------------------
    private fun buildPromptInfo(reference: FloatArray, refSampleRate: Int, refTranscript: String): PromptInfo {
        val resampled = WindowedSincResampler.resample(reference, refSampleRate, DspConstants.SR)
        val rmsResult: RmsResult = RmsNormalizer.apply(resampled)

        val promptLogMel = frontend.computeLogMel(rmsResult.signal)
        val promptLen = promptLogMel.size / DspConstants.N_MELS
        require(promptLen > 0) { "Reference produced no mel frames" }
        val promptFeatures = FloatArray(promptLogMel.size)
        for (i in promptLogMel.indices) promptFeatures[i] = promptLogMel[i] * DspConstants.FEAT_SCALE

        // Reference transcript must also tokenize.
        val promptIds = tokenizer.toIds(tokenizer.addPunctuation(refTranscript))
        require(promptIds.isNotEmpty()) { "EMPTY_TRANSCRIPT" }

        return PromptInfo(promptFeatures, promptIds, promptLen, rmsResult)
    }

    // ---------------------------------------------------------------------
    // Single-sentence synthesis (steps 3-9).
    // ---------------------------------------------------------------------
    private fun synthesizeChunk(
        targetText: String,
        prompt: PromptInfo,
        steps: Int,
        guidance: Float,
        speed: Float,
        onProgress: (done: Int, total: Int) -> Unit
    ): FloatArray {
        val promptLen = prompt.promptLen
        val promptFeatures = prompt.promptFeatures

        // 3) Tokenize.
        val targetIds = tokenizer.toIds(tokenizer.addPunctuation(targetText))
        require(targetIds.isNotEmpty()) { "NO_TOKENS" }

        // 4) Text encoder -> text_condition [1, T, 100].
        val textCondition = engine.textEncoder(targetIds, prompt.promptTokenIds, promptLen, speed)
        val totalFrames = textCondition.size / DspConstants.N_MELS
        require(totalFrames > promptLen) { "REF_LONGER" }

        // 5) Initial noise + speech conditioning.
        val rng = BoxMuller(DspConstants.SEED)
        val x = rng.nextNormals(totalFrames * DspConstants.N_MELS)
        val speechCondition = FloatArray(totalFrames * DspConstants.N_MELS)
        System.arraycopy(promptFeatures, 0, speechCondition, 0, promptFeatures.size)

        // 6) Euler flow-matching sampling.
        val y = flowSample(x, textCondition, speechCondition, steps, guidance, onProgress)

        // 7) Drop the prompt frames, transpose (keep,100)->(100,keep), unscale.
        val keep = totalFrames - promptLen
        val mels = FloatArray(DspConstants.N_MELS * keep)
        val inv = 1f / DspConstants.FEAT_SCALE   // unscale back to raw log-mel space
        for (frame in 0 until keep) {
            val src = (frame + promptLen) * DspConstants.N_MELS
            for (m in 0 until DspConstants.N_MELS) {
                // mel-channel-major layout: mels[m*keep + frame]
                mels[m * keep + frame] = y[src + m] * inv
            }
        }

        // 8) Vocos vocoder.
        val audio = vocoder.synthesize(mels, keep)

        // 9) Undo any RMS boost (only when it was applied), then clip.
        return if (prompt.rmsResult.boosted) {
            val f = prompt.rmsResult.originalRms / DspConstants.TARGET_RMS
            applyAndClip(RmsNormalizer.scale(audio, f))
        } else {
            applyAndClip(audio)
        }
    }

    private fun flowSample(
        x: FloatArray,
        textCondition: FloatArray,
        speechCondition: FloatArray,
        steps: Int,
        guidance: Float,
        onProgress: (Int, Int) -> Unit
    ): FloatArray {
        val totalFrames = textCondition.size / DspConstants.N_MELS
        // t_i = t_shift*u_i / (1 + (t_shift-1)*u_i), u_i = i/steps
        val t = FloatArray(steps + 1)
        for (i in 0..steps) {
            val u = i.toFloat() / steps
            t[i] = (DspConstants.T_SHIFT * u) / (1f + (DspConstants.T_SHIFT - 1f) * u)
        }
        for (i in 0 until steps) {
            val dt = t[i + 1] - t[i]
            val vOut = engine.fmDecoder(t[i], x, textCondition, speechCondition, guidance)
            for (j in x.indices) {
                x[j] = x[j] + vOut[j] * dt
            }
            onProgress(i + 1, steps)
        }
        return x
    }

    private fun applyAndClip(audio: FloatArray): FloatArray {
        for (i in audio.indices) {
            if (audio[i] > 1f) audio[i] = 1f
            else if (audio[i] < -1f) audio[i] = -1f
        }
        return audio
    }

    private fun concat(parts: List<FloatArray>): FloatArray {
        var total = 0
        for (p in parts) total += p.size
        val out = FloatArray(total)
        var pos = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, pos, p.size)
            pos += p.size
        }
        return out
    }

    companion object {
        // Devanagari danda (।), double danda (॥) + common sentence terminators.
        private val SENTENCE_BREAKS = charArrayOf('।', '॥', '.', '!', '?', '；', '：', '。', '！', '？')

        /**
         * Split text into sentence-sized chunks, keeping the punctuation delimiter
         * attached to the preceding sentence. Runs of whitespace are trimmed.
         * A single very long sentence with no terminator is returned as-is.
         */
        fun splitSentences(text: String): List<String> {
            val trimmed = text.trim()
            if (trimmed.isEmpty()) return emptyList()
            val chunks = ArrayList<String>()
            val current = StringBuilder()
            for (ch in trimmed) {
                current.append(ch)
                if (ch in SENTENCE_BREAKS) {
                    var s = current.toString().trim()
                    if (s.isNotEmpty()) chunks.add(s)
                    current.setLength(0)
                }
            }
            val rest = current.toString().trim()
            if (rest.isNotEmpty()) chunks.add(rest)
            return chunks
        }
    }
}

/** Minimal LCG + Box-Muller normal generator, seeded for reproducibility (seed 666). */
class BoxMuller(seed: Long) {
    private var state = seed
    private var cached: Float? = null

    private fun nextUniform(): Double {
        state = state * 6364136223846793005L + 1442695040888963407L
        val bits = (state ushr 11).toDouble()
        val v = bits / (1L shl 53).toDouble()
        return if (v <= 0.0) 1e-12 else v.coerceIn(1e-12, 1.0 - 1e-12)
    }

    fun nextFloat(): Float {
        val c = cached
        if (c != null) { cached = null; return c }
        val u1 = nextUniform()
        val u2 = nextUniform()
        val r = sqrt(-2.0 * ln(u1))
        val z0 = r * cos(2.0 * Math.PI * u2)
        val z1 = r * sin(2.0 * Math.PI * u2)
        cached = z1.toFloat()
        return z0.toFloat()
    }

    fun nextNormals(n: Int): FloatArray {
        val out = FloatArray(n)
        for (i in 0 until n) out[i] = nextFloat()
        return out
    }
}
