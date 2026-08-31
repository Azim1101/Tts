package com.swara.app.model

import com.swara.app.dsp.DspConstants

/** One downloadable artifact inside a Hugging Face model repo. */
data class ModelFileSpec(
    val name: String,
    val minBytes: Long
)

/**
 * A user-selectable TTS model package.
 *
 * The APK is intentionally small: model weights are streamed from Hugging Face
 * into the app's private storage on first use. The small `.npz` feature files
 * are converted to the app's raw `.bin` layout on-device (no Python / NumPy).
 */
data class ModelSpec(
    val id: String,
    val title: String,
    val description: String,
    val repo: String,
    val files: List<ModelFileSpec>
) {
    val baseUrl: String get() = "https://huggingface.co/$repo/resolve/main"
    fun file(name: String): ModelFileSpec? = files.firstOrNull { it.name == name }

    override fun toString(): String = title
}

object ModelCatalog {

    /**
     * MNN-only runtime path. Downloads the real `.mnn` graphs from
     * `Bbkblo/DhVaani-0.5-MNN` and runs them through the MNN Android SDK.
     *
     * The MNN runtime (`libMNN.so` + `libmnncore.so`) is bundled at build time
     * by `scripts/setup_mnn_runtime.sh`, NOT committed to Git.
     */
    val DHVAANI_MNN = ModelSpec(
        id = "dhvaani_mnn",
        title = "DhVaani 0.5 — MNN",
        description = "On-device Hindi/Indic voice clone. Runs through MNN for the fastest locally supported path.",
        repo = "Bbkblo/DhVaani-0.5-MNN",
        files = listOf(
            ModelFileSpec(DspConstants.MNN_ENCODER_INT8, 5_000_000),
            ModelFileSpec(DspConstants.MNN_FM_DECODER_INT8, 110_000_000),
            ModelFileSpec(DspConstants.MNN_VOCODER_BACKBONE, 30_000_000),
            ModelFileSpec(DspConstants.MEL_FB_NPZ, 150_000),
            ModelFileSpec(DspConstants.VOCOS_HEAD_NPZ, 1_000_000),
            ModelFileSpec(DspConstants.TOKENS_TXT, 1_000),
            // Config file present in the MNN repo; small but useful to validate.
            ModelFileSpec(DspConstants.MODEL_JSON, 100)
        )
    )

    /** Only the MNN option is offered in this build. */
    val ALL: List<ModelSpec> = listOf(DHVAANI_MNN)

    fun byId(id: String?): ModelSpec = ALL.firstOrNull { it.id == id } ?: DHVAANI_MNN
}
