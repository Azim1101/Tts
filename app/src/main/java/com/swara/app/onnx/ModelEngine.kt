package com.swara.app.onnx

/**
 * Minimal graph interface needed by the Swara synthesis pipeline.
 *
 * [MnnEngine] implements it so [com.swara.app.tts.Synthesizer] stays
 * engine-agnostic. All methods are called from the single background executor
 * owned by MainActivity.
 */
interface ModelEngine : AutoCloseable {

    /** text -> text_condition [1,T,100]. */
    fun textEncoder(
        tokens: LongArray,
        promptTokens: LongArray,
        promptFeaturesLen: Int,
        speed: Float
    ): FloatArray

    /** flow-matching step: (t, x, text_condition, speech_condition, guidance) -> v [1,T,100]. */
    fun fmDecoder(
        t: Float,
        x: FloatArray,
        textCondition: FloatArray,
        speechCondition: FloatArray,
        guidanceScale: Float
    ): FloatArray

    /** mels[1,100,T] -> vocoder hidden [1,T,512]. */
    fun vocoderBackbone(mels: FloatArray): FloatArray

    override fun close()
}
