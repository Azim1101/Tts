package com.dhvaani.app.onnx

import android.os.Build
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

/**
 * Loads the MNN native libs once (via MNNNetNative's static initializer) and
 * remembers the result.
 *
 * IMPORTANT: the previous implementation checked `Class.forName("MNNNetInstance")`,
 * which NEVER triggered the native load — the `.so` libraries are loaded inside
 * `com.taobao.android.mnn.MNNNetNative`'s static block, and merely initializing
 * `MNNNetInstance` does not initialise `MNNNetNative`. So `isRuntimeAvailable()`
 * always returned true and the app proceeded to a confusing, cryptic
 * `com.taobao.android.mnn.MNNNetNative` crash as soon as the first native call
 * tried to load the libraries. We now force-load `MNNNetNative` directly so a
 * missing/broken runtime is detected up-front and reported with the real reason
 * (including the device ABI, since this APK ships arm64-v8a only).
 */
private object MnnNative {
    private var initialized = false
    private var ok = false
    private var lastError: Throwable? = null

    fun available(): Boolean {
        return try {
            ensureAvailable()
            true
        } catch (t: Throwable) {
            false
        }
    }

    fun ensureAvailable() {
        if (initialized) {
            if (ok) return
            throw missingError()
        }
        synchronized(this) {
            if (initialized) {
                if (ok) return
                throw missingError()
            }
            try {
                // Force MNNNetNative's static block to run (loads libMNN.so,
                // libMNN_Vulkan.so, libMNN_CL.so, libMNN_GL.so, libmnncore.so).
                Class.forName("com.taobao.android.mnn.MNNNetNative")
                initialized = true
                ok = true
            } catch (t: Throwable) {
                initialized = true
                ok = false
                lastError = t
                throw missingError()
            }
        }
    }

    private fun missingError(): IllegalStateException {
        val cause = lastError
        return if (cause != null) {
            IllegalStateException(describe(), cause)
        } else {
            IllegalStateException(describe())
        }
    }

    private fun describe(): String {
        val abis = try {
            Build.SUPPORTED_ABIS.joinToString(", ")
        } catch (t: Throwable) {
            "unknown"
        }
        val cause = lastError
        val detail = when {
            cause == null -> ""
            cause.message != null -> "; ${cause.message}"
            cause.cause?.message != null -> "; ${cause.cause.message}"
            else -> "; ${cause.javaClass.simpleName}"
        }
        return "$MSG_MISSING (device abi: $abis$detail)"
    }

    private const val MSG_MISSING = "MNN_RUNTIME_MISSING"
}
