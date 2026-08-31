package com.dhvaani.app.onnx

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import com.dhvaani.app.dsp.DspConstants
import com.dhvaani.app.model.ModelSpec
import com.dhvaani.app.tts.Tokenizer
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * Resolves and materialises the selected model's engine graphs.
 *
 * The app favours a **small APK**: model weights are downloaded from Hugging Face
 * into `filesDir/models/` on first use. Bundled `assets/` are still honoured for
 * a partially-offline/bundled build, but they are optional.
 *
 * Resolution order per engine graph: `filesDir/models` first, then APK assets
 * (for an optional bundled build).
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
     * Resolve the engine graphs for [spec].
     *
     * @return ready=true with absolute on-disk paths when all graphs can be
     *         found in filesDir/models or copied from assets; ready=false with a
     *         human-readable message otherwise.
     */
    fun prepare(spec: ModelSpec): Models {
        if (!modelsDir.exists()) modelsDir.mkdirs()

        val (encNames, fmNames, vocNames) = graphCandidates(spec)

        val encoderName = resolveGraph(encNames)
        val fmName = resolveGraph(fmNames)
        val vocoderName = resolveGraph(vocNames)

        if (encoderName == null || fmName == null || vocoderName == null) {
            val missing = listOfNotNull(
                encoderName?.let { null } ?: encNames.first(),
                fmName?.let { null } ?: fmNames.first(),
                vocoderName?.let { null } ?: vocNames.first()
            )
            Log.w(TAG, "prepare(${spec.id}): missing $missing")
            return Models(
                ready = false,
                message = "Model ${spec.title} is not downloaded. Tap \"Download models\".",
                encoderPath = "", fmDecoderPath = "", vocoderBackbonePath = ""
            )
        }

        val encoderFile = materialise(encoderName)
        val fmFile = materialise(fmName)
        val backboneFile = materialise(vocoderName)

        if (encoderFile == null || fmFile == null || backboneFile == null) {
            Log.e(TAG, "prepare(${spec.id}): materialise failed (encoder=$encoderFile fm=$fmFile vocoder=$backboneFile)")
            return Models(false, "Failed to extract model files.", "", "", "")
        }
        Log.i(TAG, "prepare(${spec.id}): OK encoder=${encoderFile.absolutePath} fm=${fmFile.absolutePath} vocoder=${backboneFile.absolutePath}")
        return Models(
            true, "ok",
            encoderFile.absolutePath, fmFile.absolutePath, backboneFile.absolutePath
        )
    }

    /** Potential graph file names for a model spec, best-first. */
    private fun graphCandidates(spec: ModelSpec): Triple<Array<String>, Array<String>, Array<String>> {
        // MNN-only build.
        return Triple(
            arrayOf(DspConstants.MNN_ENCODER_INT8),
            arrayOf(DspConstants.MNN_FM_DECODER_INT8),
            arrayOf(DspConstants.MNN_VOCODER_BACKBONE)
        )
    }

    /** First candidate present in filesDir/models or assets, or null. */
    private fun resolveGraph(candidates: Array<String>): String? {
        for (name in candidates) {
            if (File(modelsDir, name).length() > 0L || assetExists(name)) return name
        }
        return null
    }

    /** Copy an asset (if needed) or reuse the downloaded file in filesDir/models. */
    private fun materialise(name: String): File? {
        val target = File(modelsDir, name)
        if (target.length() > 0L) return target
        if (!assetExists(name)) return null
        return copyAssetToModels(name)
    }

    private fun copyAssetToModels(assetName: String): File? {
        val target = File(modelsDir, assetName)
        return try {
            assets.open(assetName).use { input ->
                FileOutputStream(target).use { out -> input.copyTo(out) }
            }
            target
        } catch (e: Exception) {
            Log.e(TAG, "copyAssetToModels failed for $assetName", e)
            null
        }
    }

    private fun assetExists(name: String): Boolean {
        return try {
            val list: Array<String>? = assets.list("")
            list?.any { it == name } == true
        } catch (e: Exception) {
            false
        }
    }

    fun readAsset(name: String): ByteArray = assets.open(name).use { it.readBytes() }

    /** Read a small file from filesDir/models if present, else from assets. */
    private fun readFileOrAsset(name: String): ByteArray {
        val file = File(modelsDir, name)
        if (file.length() > 0L) {
            return FileInputStream(file).use { it.readBytes() }
        }
        return readAsset(name)
    }

    fun loadMelFb(): MelFilterbank {
        val bytes = readFileOrAsset(DspConstants.MEL_FB_BIN)
        return MelFilterbank.fromBytes(ByteArrayInputStream(bytes))
    }

    fun loadVocosHead(): VocosHead {
        val bytes = readFileOrAsset(DspConstants.VOCOS_HEAD_BIN)
        return VocosHead.fromBytes(ByteArrayInputStream(bytes))
    }

    fun loadTokenizer(): Tokenizer {
        val text = readFileOrAsset(DspConstants.TOKENS_TXT).toString(Charsets.UTF_8)
        return Tokenizer.fromFileContent(text)
    }

    private companion object {
        private const val TAG = "DhVaani.Model"
    }
}
