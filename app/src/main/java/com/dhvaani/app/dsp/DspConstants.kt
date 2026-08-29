package com.dhvaani.app.dsp

/**
 * All DSP constants shared by the frontend (log-mel), the flow-matching
 * decoder and the Vocos vocoder. These MUST stay in sync with the PyTorch
 * reference (ZipVoice / charactr-vocos-mel-24khz).
 */
object DspConstants {
    const val SR = 24000
    const val N_FFT = 1024
    const val HOP = 256
    const val N_MELS = 100
    const val FEAT_SCALE = 0.1f
    const val TARGET_RMS = 0.1f

    /** Number of frequency bins returned by rfft(1024) = 512 + 1. */
    const val N_FREQ = N_FFT / 2 + 1

    /** log10(clip(mel,1e-7,inf)) uses natural log in the reference. */
    const val MEL_MIN = 1e-7f

    // ---- Vocos head dimensions ----
    const val HIDDEN_DIM = 512
    const val OUT_DIM = 1026      // 513 magnitude + 513 phase

    // ---- Model file names (inside assets/ and filesDir/models/) ----
    const val ENCODER_INT8 = "text_encoder_int8.onnx"
    const val ENCODER_FP32 = "text_encoder.onnx"
    const val FM_DECODER_INT8 = "fm_decoder_int8.onnx"
    const val FM_DECODER_FP32 = "fm_decoder.onnx"
    const val VOCODER_BACKBONE = "vocoder_backbone.onnx"
    const val VOCOS_HEAD_BIN = "vocos_head.bin"
    const val MEL_FB_BIN = "mel_fb.bin"
    const val TOKENS_TXT = "tokens.txt"

    const val SEED = 666L
    const val T_SHIFT = 0.5f
}
