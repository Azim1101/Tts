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
 * Models live either in the bundled APK assets (fast path) or in
 * `filesDir/models/` (downloaded on first launch via [ModelDownloader]). This
 * class copies bundled assets out to `filesDir/models/`, reports what is missing,
 * and loads the small feature files from wherever they exist.
 *
 * int8 graphs are preferred, with an automatic fallback to the fp32 variants
 * (`text_encoder.onnx` / `fm_decoder.onnx`) when the int8 files are absent.
 */
class ModelManager(private val context: Context) {

    data class Models(
        val ready: Boolean,
        val message: String,
        val missing: List<String> = emptyList(),
        val encoderPath: String = "",
        val fmDecoderPath: String = "",
        val vocoderBackbonePath: String = ""
    )

    private val assets: AssetManager get() = context.assets
    val modelsDir: File get() = File(context.filesDir, "models")

    // Prefer the int8 graphs; fall back to fp32 if only those are available; if
    // neither is present we default to int8 (which the downloader fetches).
    private val encoderAsset: String
        get() = when {
            presentAssetsOrDir(DspConstants.ENCODER_INT8) -> DspConstants.ENCODER_INT8
            presentAssetsOrDir(DspConstants.ENCODER_FP32) -> DspConstants.ENCODER_FP32
            else -> DspConstants.ENCODER_INT8
        }
    private val fmAsset: String
        get() = when {
            presentAssetsOrDir(DspConstants.FM_DECODER_INT8) -> DspConstants.FM_DECODER_INT8
            presentAssetsOrDir(DspConstants.FM_DECODER_FP32) -> DspConstants.FM_DECODER_FP32
            else -> DspConstants.FM_DECODER_INT8
        }

    private fun presentAssetsOrDir(name: String): Boolean {
        return assetExists(name) || File(modelsDir, name).length() > 0L
    }

    /** Files the app must have on disk (paths under [modelsDir]) for the pipeline to run. */
    private fun requiredFiles(): List<String> = listOf(
        DspConstants.MEL_FB_BIN,
        DspConstants.VOCOS_HEAD_BIN,
        DspConstants.TOKENS_TXT,
        encoderAsset,
        fmAsset,
        DspConstants.VOCODER_BACKBONE
    )

    /**
     * Ensure the required files are present. Bundled assets are copied to
     * `filesDir/models/` (so the [OnnxEngine] can memory-map them). Returns
     * [Models] describing what is present/missing.
     */
    fun prepare(): Models {
        if (!modelsDir.exists()) modelsDir.mkdirs()

        // Copy any bundled assets into modelsDir (idempotent).
        for (name in setOf(encoderAsset, fmAsset, DspConstants.VOCODER_BACKBONE,
                DspConstants.MEL_FB_BIN, DspConstants.VOCOS_HEAD_BIN, DspConstants.TOKENS_TXT)) {
            val target = File(modelsDir, name)
            if (target.length() == 0L && assetExists(name)) {
                copyAssetTo(name, target)
            }
        }

        // int8 -> fp32 fallback resolved above; now collect what's still missing.
        val missing = requiredFiles().filter { File(modelsDir, it).length() == 0L }
        if (missing.isNotEmpty()) {
            return Models(
                ready = false,
                message = "Model files missing: $missing. Download them or run scripts/download_models.sh.",
                missing = missing
            )
        }
        return Models(
            ready = true,
            message = "ok",
            encoderPath = File(modelsDir, encoderAsset).absolutePath,
            fmDecoderPath = File(modelsDir, fmAsset).absolutePath,
            vocoderBackbonePath = File(modelsDir, DspConstants.VOCODER_BACKBONE).absolutePath
        )
    }

    /**
     * Download whatever is still needed into [modelsDir] (streaming, with retry +
     * size check), then regenerate the `.bin` files. Returns true if the app is
     * ready afterwards.
     */
    fun downloadMissingModels(progress: ModelDownloader.Progress): Boolean {
        val current = prepare()
        if (current.ready) return true

        val needed = ArrayList<String>()
        fun addForOnnx(asset: String) { if (File(modelsDir, asset).length() == 0L) needed.add(asset) }
        addForOnnx(encoderAsset)
        addForOnnx(fmAsset)
        addForOnnx(DspConstants.VOCODER_BACKBONE)
        if (File(modelsDir, DspConstants.MEL_FB_BIN).length() == 0L) needed.add("mel_fb.npz")
        if (File(modelsDir, DspConstants.VOCOS_HEAD_BIN).length() == 0L) needed.add("vocos_head.npz")
        if (File(modelsDir, DspConstants.TOKENS_TXT).length() == 0L) needed.add(DspConstants.TOKENS_TXT)

        if (needed.isEmpty()) return prepare().ready

        val ok = ModelDownloader.downloadMissing(modelsDir, needed, progress)
        return ok && prepare().ready
    }

    private fun copyAssetTo(name: String, target: File) {
        try {
            assets.open(name).use { input ->
                FileOutputStream(target).use { out -> input.copyTo(out) }
            }
        } catch (e: Exception) {
            // ignored; prepare() reports missing
        }
    }

    private fun assetExists(name: String): Boolean {
        return try {
            assets.list("")?.any { it == name } == true
        } catch (e: Exception) {
            false
        }
    }

    private fun readSource(name: String): ByteArray {
        val f = File(modelsDir, name)
        if (f.length() > 0) return f.readBytes()
        return assets.open(name).use { it.readBytes() }
    }

    fun readAsset(name: String): ByteArray = assets.open(name).use { it.readBytes() }

    fun loadMelFb(): MelFilterbank {
        val bytes = readSource(DspConstants.MEL_FB_BIN)
        return MelFilterbank.fromBytes(ByteArrayInputStream(bytes))
    }

    fun loadVocosHead(): VocosHead {
        val bytes = readSource(DspConstants.VOCOS_HEAD_BIN)
        return VocosHead.fromBytes(ByteArrayInputStream(bytes))
    }

    fun loadTokenizer(): Tokenizer {
        val text = readSource(DspConstants.TOKENS_TXT).toString(Charsets.UTF_8)
        return Tokenizer.fromFileContent(text)
    }
}
