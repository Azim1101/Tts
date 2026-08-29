package com.dhvaani.app.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Thin wrapper around onnxruntime for the three ONNX graphs. Sessions are created
 * once with `ALL_OPT` and run repeatedly. It is important NOT to close a session
 * inside a run — sessions are closed in [close].
 *
 * Thread-safety note: ORT sessions are thread-safe for Run, but each [OnnxTensor]
 * must be created on the calling thread. We therefore keep everything on a single
 * background executor inside [com.dhvaani.app.tts.Synthesizer].
 */
class OnnxEngine(
    private val encoderPath: String,
    private val fmDecoderPath: String,
    private val vocoderBackbonePath: String,
    private val threads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val encoder: OrtSession
    private val fmDecoder: OrtSession
    private val vocoderBackbone: OrtSession

    init {
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.setIntraOpNumThreads(threads)
        opts.setInterOpNumThreads(threads)

        encoder = env.createSession(encoderPath, opts)
        fmDecoder = env.createSession(fmDecoderPath, opts)
        vocoderBackbone = env.createSession(vocoderBackbonePath, opts)
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
        val (tShape, tTensor) = longTensor1D(tokens)
        val (pShape, pTensor) = longTensor1D(promptTokens)
        val (lenShape, lenTensor) = longScalar(promptFeaturesLen.toLong())
        val (spShape, spTensor) = floatScalar(speed)

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
        val (tShape, tTensor) = floatScalar(t)
        val (xShape, xTensor) = floatTensor3D(x, length)
        val (tcShape, tcTensor) = floatTensor3D(textCondition, length)
        val (scShape, scTensor) = floatTensor3D(speechCondition, length)
        val (gShape, gTensor) = floatScalar(guidanceScale)

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
    private fun floatScalar(value: Float): Pair<LongArray, OnnxTensor> {
        return longArrayOf() to OnnxTensor.createTensor(env, directFloat(floatArrayOf(value)), longArrayOf())
    }

    private fun longScalar(value: Long): Pair<LongArray, OnnxTensor> {
        return longArrayOf() to OnnxTensor.createTensor(env, directLong(longArrayOf(value)), longArrayOf())
    }

    private fun floatTensor3D(data: FloatArray, length: Int): Pair<LongArray, OnnxTensor> {
        val shape = longArrayOf(1L, length.toLong(), 100L)
        return shape to OnnxTensor.createTensor(env, directFloat(data), shape)
    }

    private fun longTensor1D(data: LongArray): Pair<LongArray, OnnxTensor> {
        val shape = longArrayOf(1L, data.size.toLong())
        return shape to OnnxTensor.createTensor(env, directLong(data), shape)
    }

    /**
     * Wrap a FloatArray into a native-order direct [FloatBuffer]. We build it from
     * a direct [ByteBuffer] so ORT can read it directly and we never depend on the
     * (platform-dependent) return type of `FloatBuffer.order(ByteOrder)`.
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
}
