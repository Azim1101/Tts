package com.dhvaani.app.onnx

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.dhvaani.app.dsp.DspConstants
import com.dhvaani.app.tts.Tokenizer
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream

/**
 * Resolves and materialises the model assets.
 *
 * This is the **v0.5.0-style** simple prepare: it copies any model files
 * that exist in APK `assets/` (or in `filesDir/models/` from a prior run)
 * and reports the on-disk paths. There is no `setOf`/`filter` chain, no
 * minBytes thresholds, and no read-side `for (name in setOf(...))` — the
 * v0.7 `setOf(encoderAsset, fmAsset, …)` getter evaluation in `prepare()`
 * was the most likely site of the `List.iterator()` NPE the user reported.
 *
 * The (large) `.onnx` graphs are copied once to `filesDir/models/` so
 * onnxruntime can memory-map them directly; on subsequent launches the
 * existing copy is reused.
 *
 * int8 graphs are preferred, with an automatic fallback to the fp32
 * variants (`text_encoder.onnx` / `fm_decoder.onnx`) when the int8 files
 * are absent.
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

    /** `filesDir/models/` — created lazily. */
    private val modelsDir: File get() = File(context.filesDir, "models")

    /**
     * Make sure the required files are present and report the on-disk paths.
     *
     * Logic (mirrors v0.5.0, which the user confirmed was working):
     * 1. Pick int8 file names if present, else fall back to fp32 names.
     * 2. If any of the chosen names is missing from BOTH assets and
     *    filesDir/models, return ready=false with a useful message.
     * 3. Otherwise copy from assets to filesDir (idempotent — skips files
     *    that already exist on disk with non-zero size) and return ready=true
     *    with the absolute paths.
     */
    fun prepare(): Models {
        if (!modelsDir.exists()) modelsDir.mkdirs()

        val encoderAsset =
            if (assetExists(DspConstants.ENCODER_INT8)) DspConstants.ENCODER_INT8
            else DspConstants.ENCODER_FP32
        val fmAsset =
            if (assetExists(DspConstants.FM_DECODER_INT8)) DspConstants.FM_DECODER_INT8
            else DspConstants.FM_DECODER_FP32
        val vocoderAsset = DspConstants.VOCODER_BACKBONE

        if (!assetExists(vocoderAsset) || !assetExists(encoderAsset) || !assetExists(fmAsset)) {
            Log.w(TAG, "prepare: at least one model missing from assets (encoder=$encoderAsset fm=$fmAsset vocoder=$vocoderAsset)")
            return Models(
                ready = false,
                message = "Model assets missing. Run scripts/download_models.sh before building, or use the in-app download.",
                encoderPath = "", fmDecoderPath = "", vocoderBackbonePath = ""
            )
        }

        val encoderFile = copyToModels(encoderAsset)
        val fmFile = copyToModels(fmAsset)
        val backboneFile = copyToModels(vocoderAsset)

        if (encoderFile == null || fmFile == null || backboneFile == null) {
            Log.e(TAG, "prepare: copyToModels failed (encoder=$encoderFile fm=$fmFile vocoder=$backboneFile)")
            return Models(false, "Failed to extract model files.", "", "", "")
        }
        Log.i(TAG, "prepare: OK encoder=${encoderFile.absolutePath} fm=${fmFile.absolutePath} vocoder=${backboneFile.absolutePath}")
        return Models(
            true, "ok",
            encoderFile.absolutePath, fmFile.absolutePath, backboneFile.absolutePath
        )
    }

    /** Copy a single asset to `filesDir/models/` if it's not already there. */
    private fun copyToModels(assetName: String): File? {
        val target = File(modelsDir, assetName)
        if (target.length() > 0L) return target  // already extracted
        return try {
            assets.open(assetName).use { input ->
                FileOutputStream(target).use { out -> input.copyTo(out) }
            }
            target
        } catch (e: Exception) {
            Log.e(TAG, "copyToModels: failed for $assetName", e)
            null
        }
    }

    private fun assetExists(name: String): Boolean {
        return try {
            // AssetManager.list(path) returns the names in the given asset
            // directory; on the root we pass "". The result is nullable on
            // some Android versions, hence the safe-call.
            val list: Array<String>? = assets.list("")
            list?.any { it == name } == true
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

    private companion object {
        private const val TAG = "DhVaani.Model"
    }
}
