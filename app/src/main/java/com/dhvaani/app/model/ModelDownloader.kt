package com.dhvaani.app.model

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HttpsURLConnection
import zone.dhvaani.tts.DhVaani

/**
 * Downloads the DhVaani-0.5 MNN models and feature assets directly from Hugging Face
 * onto the device storage on first launch (if not bundled in assets).
 */
object ModelDownloader {

    private const val BASE = "https://huggingface.co/Bbkblo/DhVaani-0.5-MNN/resolve/main"
    private const val MAX_ATTEMPTS = 5

    data class Spec(val assetName: String, val remoteUrl: String, val minBytes: Long)

    val SPECS: List<Spec> = listOf(
        Spec("text_encoder_int8.mnn", "$BASE/text_encoder_int8.mnn", 5_000_000L),
        Spec("fm_decoder_int8.mnn", "$BASE/fm_decoder_int8.mnn", 110_000_000L),
        Spec("vocoder_backbone.mnn", "$BASE/vocoder_backbone.mnn", 12_000_000L),
        Spec("mel_fb.bin", "$BASE/mel_fb.bin", 200_000L),
        Spec("vocos_head.bin", "$BASE/vocos_head.bin", 2_000_000L),
        Spec("tokens.txt", "$BASE/tokens.txt", 1_000L),
        Spec("model.json", "$BASE/model.json", 500L),
    )

    /** Progress callback: (fileIndex 1-based, fileCount, fileName, fraction 0..1). */
    fun interface Progress {
        fun onProgress(fileIndex: Int, fileCount: Int, fileName: String, fraction: Float)
    }

    /**
     * Downloads missing models into [dir]. Returns true if all [DhVaani.REQUIRED_FILES] are present.
     */
    fun downloadMissing(dir: File, progress: Progress): Boolean {
        if (!dir.exists()) dir.mkdirs()
        val toFetch = SPECS.filter { File(dir, it.assetName).length() < it.minBytes }
        val totalFiles = toFetch.size
        toFetch.forEachIndexed { idx, spec ->
            val target = File(dir, spec.assetName)
            if (!downloadFile(spec.remoteUrl, target, spec.minBytes) { f ->
                    progress.onProgress(idx + 1, totalFiles, spec.assetName, f)
                }
            ) {
                return false
            }
            progress.onProgress(idx + 1, totalFiles, spec.assetName, 1f)
        }
        return DhVaani.REQUIRED_FILES.all { File(dir, it).length() > 0 }
    }

    private fun downloadFile(url: String, dest: File, minBytes: Long, onByte: (Float) -> Unit): Boolean {
        val part = File(dest.parentFile, dest.name + ".part")
        for (attempt in 1..MAX_ATTEMPTS) {
            var conn: HttpURLConnection? = null
            try {
                part.delete()
                conn = (URL(url).openConnection() as HttpsURLConnection).apply {
                    connectTimeout = 60_000
                    readTimeout = 180_000
                    instanceFollowRedirects = true
                    requestMethod = "GET"
                    setRequestProperty("User-Agent", "DhVaani-Android/1.0")
                }
                val code = conn.responseCode
                if (code !in 200..299) {
                    conn.disconnect()
                    continue
                }
                val totalLen = conn.contentLengthLong
                var total: Long = 0
                conn.inputStream.use { input ->
                    FileOutputStream(part).use { out ->
                        val buf = ByteArray(64 * 1024)
                        var n = input.read(buf)
                        while (n > 0) {
                            out.write(buf, 0, n)
                            total += n
                            if (totalLen > 0) onByte(total.toFloat() / totalLen.toFloat())
                            n = input.read(buf)
                        }
                    }
                }
                if (part.length() >= minBytes) {
                    part.renameTo(dest)
                    return true
                }
            } catch (e: IOException) {
                // retry
            } finally {
                try { conn?.disconnect() } catch (_: Exception) {}
            }
        }
        part.delete()
        return false
    }
}
