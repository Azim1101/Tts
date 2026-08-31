package com.dhvaani.app.onnx

import android.util.Log
import com.taobao.android.mnn.MNNForwardType
import com.taobao.android.mnn.MNNNetInstance

/**
 * MNN backend for the DhVaani `.mnn` graphs
 * (`text_encoder_int8.mnn`, `fm_decoder_int8.mnn`, `vocoder_backbone.mnn`),
 * downloaded from `Bbkblo/DhVaani-0.5-MNN`.
 *
 * The MNN Android runtime (`libMNN.so` + `libmnncore.so`) is fetched at build
 * time by `scripts/setup_mnn_runtime.sh` into `app/src/main/jniLibs/arm64-v8a`.
 * The Java wrapper classes under `com.taobao.android.mnn` are part of this app,
 * so the native `libmnncore.so` links directly — no separate MNN jar needed.
 *
 * Notes:
 *  - All calls run on the single background executor owned by MainActivity.
 *  - MNN is an experimental path for these graphs; the converter notes that the
 *    graphs contain subgraphs, so if the older Interpreter/Session API cannot
 *    run them the constructor/run throws and the UI shows a clear error instead
 *    of silently producing wrong audio.
 */
class MnnEngine(
    encoderPath: String,
    fmDecoderPath: String,
    vocoderBackbonePath: String,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
) : ModelEngine {

    private val encoder: MnnNet
    private val fmDecoder: MnnNet
    private val vocoder: MnnNet

    init {
        MnnNative.ensureAvailable()
        encoder = MnnNet(encoderPath, threads, "encoder")
        fmDecoder = MnnNet(fmDecoderPath, threads, "fm_decoder")
        vocoder = MnnNet(vocoderBackbonePath, threads, "vocoder_backbone")
        Log.i(TAG, "MNN runtime loaded: encoder, fm_decoder, vocoder_backbone")
    }

    override fun textEncoder(
        tokens: LongArray,
        promptTokens: LongArray,
        promptFeaturesLen: Int,
        speed: Float
    ): FloatArray {
        val tokenInts = longArrayToIntArray(tokens)
        val promptInts = longArrayToIntArray(promptTokens)
        val ins = linkedMapOf(
            "tokens" to MnnInput.int(tokenInts, intArrayOf(1, tokenInts.size)),
            "prompt_tokens" to MnnInput.int(promptInts, intArrayOf(1, promptInts.size)),
            "prompt_features_len" to MnnInput.int(intArrayOf(promptFeaturesLen), intArrayOf(1)),
            "speed" to MnnInput.float(floatArrayOf(speed), intArrayOf(1))
        )
        return encoder.run(ins, "text_condition")
    }

    override fun fmDecoder(
        t: Float,
        x: FloatArray,
        textCondition: FloatArray,
        speechCondition: FloatArray,
        guidanceScale: Float
    ): FloatArray {
        val length = textCondition.size / 100
        val dims = intArrayOf(1, length, 100)
        val ins = linkedMapOf(
            "t" to MnnInput.float(floatArrayOf(t), intArrayOf(1)),
            "x" to MnnInput.float(x, dims),
            "text_condition" to MnnInput.float(textCondition, dims),
            "speech_condition" to MnnInput.float(speechCondition, dims),
            "guidance_scale" to MnnInput.float(floatArrayOf(guidanceScale), intArrayOf(1))
        )
        return fmDecoder.run(ins, "v")
    }

    override fun vocoderBackbone(mels: FloatArray): FloatArray {
        val frames = mels.size / 100
        return vocoder.run(
            linkedMapOf("mels" to MnnInput.float(mels, intArrayOf(1, 100, frames))),
            "hidden"
        )
    }

    override fun close() {
        runCatching { encoder.close() }
        runCatching { fmDecoder.close() }
        runCatching { vocoder.close() }
    }

    companion object {
        private const val TAG = "DhVaani.Mnn"

        /** Quick check used by the UI before offering / loading a model. */
        fun isRuntimeAvailable(): Boolean = MnnNative.available()
    }

    private class MnnInput(
        val floats: FloatArray?,
        val ints: IntArray?,
        val dims: IntArray
    ) {
        companion object {
            fun float(data: FloatArray, dims: IntArray) = MnnInput(data, null, dims)
            fun int(data: IntArray, dims: IntArray) = MnnInput(null, data, dims)
        }
    }

    /**
     * A single `.mnn` graph using the MNNNetInstance / Session / Tensor API.
     * Inputs are named (the converter preserved the graph input names); outputs
     * are read by name too.
     */
    private class MnnNet(
        path: String,
        threads: Int,
        label: String
    ) : AutoCloseable {

        private val instance: MNNNetInstance
        private val session: MNNNetInstance.Session

        init {
            instance = MNNNetInstance.createFromFile(path)
                ?: throw IllegalStateException("MNN createFromFile returned null for $path")
            val config = MNNNetInstance.Config()
            config.numThread = threads
            config.forwardType = MNNForwardType.FORWARD_CPU.type
            session = instance.createSession(config)
                ?: throw IllegalStateException("MNN createSession returned null for $label")
            Log.i(TAG, "MnnNet($label) loaded")
        }

        fun run(inputs: Map<String, MnnInput>, outputName: String?): FloatArray {
            for ((name, input) in inputs) {
                val tensor = session.getInput(name)
                    ?: throw IllegalStateException("MNN input '$name' not found")
                tensor.reshape(input.dims)
                if (input.floats != null) {
                    tensor.setInputFloatData(input.floats)
                } else {
                    tensor.setInputIntData(input.ints!!)
                }
            }
            session.reshape()
            session.run()
            val output = session.getOutput(outputName)
                ?: throw IllegalStateException("MNN output '$outputName' not found")
            return output.getFloatData()
        }

        override fun close() {
            runCatching { session.release() }
            runCatching { instance.release() }
        }
    }
}

private fun longArrayToIntArray(src: LongArray): IntArray {
    val out = IntArray(src.size)
    for (i in src.indices) out[i] = src[i].toInt()
    return out
}

/** Tries to load the MNN native libs once and remembers the result. */
private object MnnNative {
    private var initialized = false
    private var ok = false

    fun available(): Boolean {
        try {
            ensureAvailable()
            return true
        } catch (t: Throwable) {
            return false
        }
    }

    fun ensureAvailable() {
        if (initialized && ok) return
        synchronized(this) {
            if (initialized && ok) return
            if (initialized && !ok) throw IllegalStateException(MSG_MISSING)
            try {
                // The static init of MNNNetInstance loads libMNN.so + libmnncore.so.
                Class.forName("com.taobao.android.mnn.MNNNetInstance")
                initialized = true
                ok = true
            } catch (t: Throwable) {
                initialized = true
                ok = false
                throw IllegalStateException(MSG_MISSING, t)
            }
        }
    }

    private const val MSG_MISSING = "MNN_RUNTIME_MISSING"
}
