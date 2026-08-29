package com.dhvaani.app.onnx

import android.content.Context
import android.content.res.AssetManager
import com.dhvaani.app.dsp.DspConstants
import com.dhvaani.app.tts.Tokenizer
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream

/**
 * Resolves and materialises the model assets.
 *
 * The small feature files (`*.bin`, `tokens.txt`) are read straight from APK
 * assets. The (large) `.onnx` graphs are copied once to `filesDir/models/` so
 * onnxruntime can memory-map them directly; on subsequent launches the existing
 * copy is reused.
 *
 * int8 graphs are preferred, with an automatic fallback to the fp32 variants
 * (`text_encoder.onnx` / `fm_decoder.onnx`) when the int8 files are absent.
 */
class ModelManager(private val context: Context) {

    data class Models(
        val ready: Boolean,
        val message: String,
        val encoderPath: String,
        val fmDecoderPath: String,
        val vocoderBackbonePath: String
    )

    private val assets: AssetManager get() = context.assets

    fun prepare(): Models {
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) modelsDir.mkdirs()

        val encoderAsset = if (assetExists(DspConstants.ENCODER_INT8)) DspConstants.ENCODER_INT8
        else DspConstants.ENCODER_FP32
        val fmAsset = if (assetExists(DspConstants.FM_DECODER_INT8)) DspConstants.FM_DECODER_INT8
        else DspConstants.FM_DECODER_FP32
        val vocoderAsset = DspConstants.VOCODER_BACKBONE

        if (!assetExists(vocoderAsset) || !assetExists(encoderAsset) || !assetExists(fmAsset)) {
            return Models(
                ready = false,
                message = "Model assets missing (expected $encoderAsset, $fmAsset, $vocoderAsset in assets).",
                encoderPath = "", fmDecoderPath = "", vocoderBackbonePath = ""
            )
        }

        val encoderFile = copyToModels(modelsDir, encoderAsset)
        val fmFile = copyToModels(modelsDir, fmAsset)
        val backboneFile = copyToModels(modelsDir, vocoderAsset)

        if (encoderFile == null || fmFile == null || backboneFile == null) {
            return Models(false, "Failed to extract model files.", "", "", "")
        }
        return Models(true, "ok", encoderFile.absolutePath, fmFile.absolutePath, backboneFile.absolutePath)
    }

    private fun copyToModels(dir: File, assetName: String): File? {
        val target = File(dir, assetName)
        if (target.length() > 0) return target   // already extracted
        return try {
            assets.open(assetName).use { input ->
                FileOutputStream(target).use { out ->
                    input.copyTo(out)
                }
            }
            target
        } catch (e: Exception) {
            null
        }
    }

    private fun assetExists(name: String): Boolean {
        return try {
            assets.list("")?.any { it == name } == true
        } catch (e: Exception) {
            false
        }
    }

    fun readAsset(name: String): ByteArray = assets.open(name).use { it.readBytes() }

    fun loadMelFb(): MelFilterbank {
        val bytes = readAsset(DspConstants.MEL_FB_BIN)
        return MelFilterbank.fromBytes(ByteArrayInputStream(bytes))
    }

    fun loadVocosHead(): VocosHead {
        val bytes = readAsset(DspConstants.VOCOS_HEAD_BIN)
        return VocosHead.fromBytes(ByteArrayInputStream(bytes))
    }

    fun loadTokenizer(): Tokenizer {
        val text = readAsset(DspConstants.TOKENS_TXT).toString(Charsets.UTF_8)
        return Tokenizer.fromFileContent(text)
    }
}
