package com.dhvaani.app.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Thin wrapper around onnxruntime for the three ONNX graphs.
 *
 * Sessions are created once with `ALL_OPT` and run repeatedly. It is important
 * NOT to close a session inside a run — sessions are closed in [close].
 *
 * ## Execution provider policy
 *
 * We default to **plain CPU** because every Android device supports it and
 * it always produces correct results. XNNPACK is enabled on a best-effort
 * basis (HashMap, with `intra_op_num_threads`); if it fails to attach on a
 * given device we silently fall back to CPU so the app never crashes on
 * model load.
 *
 * The previous v0.6/v0.7 build tried XNNPACK -> NNAPI -> CPU with a *single
 * `SessionOptions` reused across attempts* and `addXnnpack(emptyMap())`,
 * which surfaced as a `List.iterator()` NPE on some Android 14/15 devices.
 *
 * Thread-safety note: ORT sessions are thread-safe for Run, but each
 * [OnnxTensor] must be created on the calling thread. We therefore keep
 * everything on a single background executor inside
 * [com.dhvaani.app.tts.Synthesizer].
 */
class OnnxEngine(
    private val encoderPath: String,
    private val fmDecoderPath: String,
    private val vocoderBackbonePath: String,
    // SD680 has 8 cores; SD888/8 Gen 1/2/3 have 8 too. Older devices cap at 4.
    // We allow up to 8 (the ORT practical sweet spot for int8 transformer-ish
    // graphs); the previous 4-cap was the v0.5-v0.7 bottleneck on modern SoCs.
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val fmDecoder: OrtSession
    private val vocoderBackbone: OrtSession

    init {
        Log.i(TAG, "OnnxEngine init: encoder=$encoderPath")
        Log.i(TAG, "OnnxEngine init: fmDecoder=$fmDecoderPath")
        Log.i(TAG, "OnnxEngine init: vocoderBackbone=$vocoderBackbonePath")
        encoder = createSession(encoderPath, "encoder")
        fmDecoder = createSession(fmDecoderPath, "fm_decoder")
        vocoderBackbone = createSession(vocoderBackbonePath, "vocoder_backbone")
        Log.i(TAG, "OnnxEngine init: all three sessions created")
    }

    /**
     * Create a single ORT session. We try XNNPACK first (using a fresh,
     * mutable [HashMap] of provider options, since the Kotlin
     * [kotlin.collections.emptyMap] singleton is not what ORT's JNI code
     * expects to iterate). If anything throws — option not supported on this
     * device, op not supported, NPE in the native provider list — we fall
     * back to plain CPU so the app stays usable.
     */
    private fun createSession(path: String, label: String): OrtSession {
        val t0 = SystemClock.elapsedRealtime()
        try {
            val opts = OrtSession.SessionOptions()
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(threads)
            opts.setInterOpNumThreads(threads)
            // Memory pattern optimization lets ORT reuse pre-allocated buffers
            // across inferences — the FM-decoder loop runs `steps` times, so a
            // lot of allocations can be folded.
            try { opts.setMemoryPatternOptimization(true) } catch (_: Exception) {}
            // CPU arena allocator keeps a small pool and avoids malloc/free churn.
            try { opts.setCPUArenaAllocator(true) } catch (_: Exception) {}
            val xnnOpts = HashMap<String, String>()
            xnnOpts["intra_op_num_threads"] = threads.toString()
            opts.addXnnpack(xnnOpts)
            val session = env.createSession(path, opts)
            try { opts.close() } catch (_: Exception) {}
            val ms = SystemClock.elapsedRealtime() - t0
            Log.i(TAG, "createSession($label): XNNPACK OK (threads=$threads, ${ms}ms)")
            return session
        } catch (e: Throwable) {
            Log.w(TAG, "createSession($label): XNNPACK failed (${e::class.java.simpleName}: ${e.message}); falling back to CPU")
            return createSessionCpu(path, label)
        }
    }

    /** Plain CPU session, used as the universal fallback. */
    private fun createSessionCpu(path: String, label: String): OrtSession {
        val t0 = SystemClock.elapsedRealtime()
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.setIntraOpNumThreads(threads)
        opts.setInterOpNumThreads(threads)
        try { opts.setMemoryPatternOptimization(true) } catch (_: Exception) {}
        try { opts.setCPUArenaAllocator(true) } catch (_: Exception) {}
        val session = env.createSession(path, opts)
        try { opts.close() } catch (_: Exception) {}
        val ms = SystemClock.elapsedRealtime() - t0
        Log.i(TAG, "createSession($label): CPU OK (threads=$threads, ${ms}ms)")
        return session
    }

    // ---------------------------------------------------------------------
    // text_encoder:  tokens[1,T  ], prompt_tokens[1,T'], prompt_features_len{}, speed{}
    //   -> text_condition[1,T,100]
    // ---------------------------------------------------------------------
    fun textEncoder(
        tokens: LongArray,
        promptTokens: LongArray,
        promptFeaturesLen: Int,
        speed: Float
    ): FloatArray {
        val tTensor = longTensor1D(tokens)
        val pTensor = longTensor1D(promptTokens)
        val lenTensor = longScalar(promptFeaturesLen.toLong())
        val spTensor = floatScalar(speed)

        try {
            val result = encoder.run(
                mapOf(
                    "tokens" to tTensor,
                    "prompt_tokens" to pTensor,
                    "prompt_features_len" to lenTensor,
                    "speed" to spTensor
                )
            )
            val output = result.get(0) as OnnxTensor
            val data = output.floatBuffer
            val arr = FloatArray(data.remaining())
            data.get(arr)
            result.close()
            return arr
        } finally {
            tTensor.close(); pTensor.close(); lenTensor.close(); spTensor.close()
        }
    }

    // ---------------------------------------------------------------------
    // fm_decoder:  t{}, x[1,L,100], text_condition[1,L,100], speech_condition[1,L,100],
    //              guidance_scale{}  -> v[1,L,100]
    // ---------------------------------------------------------------------
    fun fmDecoder(
        t: Float,
        x: FloatArray,
        textCondition: FloatArray,
        speechCondition: FloatArray,
        guidanceScale: Float
    ): FloatArray {
        val length = textCondition.size / 100
        val tTensor = floatScalar(t)
        val xTensor = floatTensor3D(x, length)
        val tcTensor = floatTensor3D(textCondition, length)
        val scTensor = floatTensor3D(speechCondition, length)
        val gTensor = floatScalar(guidanceScale)

        val t0 = SystemClock.elapsedRealtime()
        try {
            val result = fmDecoder.run(
                mapOf(
                    "t" to tTensor,
                    "x" to xTensor,
                    "text_condition" to tcTensor,
                    "speech_condition" to scTensor,
                    "guidance_scale" to gTensor
                )
            )
            val output = result.get(0) as OnnxTensor
            val data = output.floatBuffer
            val arr = FloatArray(data.remaining())
            data.get(arr)
            result.close()
            return arr
        } finally {
            tTensor.close(); xTensor.close(); tcTensor.close(); scTensor.close(); gTensor.close()
            val ms = SystemClock.elapsedRealtime() - t0
            if (ms > 100) Log.d(TAG, "fmDecoder: T=$length, ${ms}ms")
        }
    }

    // ---------------------------------------------------------------------
    // vocoder_backbone: mels[1,100,L] -> hidden[1,L,512]
    // ---------------------------------------------------------------------
    fun vocoderBackbone(mels: FloatArray): FloatArray {
        val frames = mels.size / 100
        val shape = longArrayOf(1L, 100L, frames.toLong())
        val tensor = OnnxTensor.createTensor(env, directFloat(mels), shape)
        try {
            val result = vocoderBackbone.run(mapOf("mels" to tensor))
            val output = result.get(0) as OnnxTensor
            val data = output.floatBuffer
            val arr = FloatArray(data.remaining())
            data.get(arr)
            result.close()
            return arr
        } finally {
            tensor.close()
        }
    }

    override fun close() {
        encoder.close()
        fmDecoder.close()
        vocoderBackbone.close()
    }

    // ---------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------
    private fun floatScalar(value: Float): OnnxTensor {
        return OnnxTensor.createTensor(env, directFloat(floatArrayOf(value)), longArrayOf())
    }

    private fun longScalar(value: Long): OnnxTensor {
        return OnnxTensor.createTensor(env, directLong(longArrayOf(value)), longArrayOf())
    }

    private fun floatTensor3D(data: FloatArray, length: Int): OnnxTensor {
        val shape = longArrayOf(1L, length.toLong(), 100L)
        return OnnxTensor.createTensor(env, directFloat(data), shape)
    }

    private fun longTensor1D(data: LongArray): OnnxTensor {
        val shape = longArrayOf(1L, data.size.toLong())
        return OnnxTensor.createTensor(env, directLong(data), shape)
    }

    /**
     * Wrap a FloatArray into a native-order direct [FloatBuffer]. We build it
     * from a direct [ByteBuffer] so ORT can read it directly and we never
     * depend on the (platform-dependent) return type of
     * `FloatBuffer.order(ByteOrder)`.
     */
    private fun directFloat(data: FloatArray): FloatBuffer {
        val bb = ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(data)
        fb.rewind()
        return fb
    }

    private fun directLong(data: LongArray): LongBuffer {
        val bb = ByteBuffer.allocateDirect(data.size * 8).order(ByteOrder.nativeOrder())
        val lb = bb.asLongBuffer()
        lb.put(data)
        lb.rewind()
        return lb
    }

    private companion object {
        private const val TAG = "DhVaani.Ort"
    }
}
