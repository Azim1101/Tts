package com.swara.app.onnx

import com.swara.app.dsp.DspConstants
import com.swara.app.model.ModelSpec
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HttpsURLConnection

/**
 * Downloads a [ModelSpec]'s graphs/features from Hugging Face directly onto the
 * device (the "first launch download" path), so the APK stays small and no model
 * files need to be committed or bundled.
 *
 * Streaming + retry + minimum-byte-size check handle the large
 * `fm_decoder_int8.onnx` / `.mnn` (~125 MB) which can truncate.
 *
 * The small feature files come as `.npz`; they are converted to the app's `.bin`
 * format here using [NpzParser] (no Python/NumPy at runtime).
 *
 * No audio ever leaves the device — this only downloads static model weights once.
 */
object ModelDownloader {

    private const val MAX_ATTEMPTS = 5

    /** Progress callback: (fileIndex 1-based, fileCount, fileName, fraction 0..1). */
    fun interface Progress {
        fun onProgress(fileIndex: Int, fileCount: Int, fileName: String, fraction: Float)
    }

    /**
     * Ensure every file in [spec] exists in [dir], downloading and then generating
     * the derived `.bin` files. Returns true when all files are present.
     */
    fun downloadMissing(dir: File, spec: ModelSpec, progress: Progress): Boolean {
        val toFetch = spec.files.filter {
            File(dir, it.name).length() < it.minBytes
        }
        val totalFiles = toFetch.size
        toFetch.forEachIndexed { idx, file ->
            val target = File(dir, file.name)
            val remote = "${spec.baseUrl}/${file.name}"
            if (!downloadFile(remote, target, file.minBytes) { f ->
                    progress.onProgress(idx + 1, totalFiles, file.name, f)
                }
            ) {
                return false
            }
            progress.onProgress(idx + 1, totalFiles, file.name, 1f)
        }
        // Convert npz -> bin (the loaders read the .bin files).
        generateBins(dir)
        // Re-check every spec file is present (and minimally sized) on disk.
        return spec.files.all {
            val f = File(dir, it.name)
            f.length() > 0L && f.length() >= it.minBytes
        }
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
                    setRequestProperty("User-Agent", "Swara-android/2.0")
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
                            if (totalLen > 0) onByte((total.toFloat() / totalLen.toFloat()))
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

    // ---------------------------------------------------------------------
    // npz -> bin generation (same layout as scripts/make_assets.py)
    // ---------------------------------------------------------------------
    @Synchronized
    fun generateBins(dir: File) {
        val melNpz = File(dir, DspConstants.MEL_FB_NPZ)
        if (melNpz.length() > 0 && !File(dir, DspConstants.MEL_FB_BIN).exists()) {
            try {
                writeMelFb(melNpz, File(dir, DspConstants.MEL_FB_BIN))
            } catch (_: Exception) {}
        }
        val headNpz = File(dir, DspConstants.VOCOS_HEAD_NPZ)
        if (headNpz.length() > 0 && !File(dir, DspConstants.VOCOS_HEAD_BIN).exists()) {
            try {
                writeVocosHead(headNpz, File(dir, DspConstants.VOCOS_HEAD_BIN))
            } catch (_: Exception) {}
        }
    }

    private fun writeMelFb(npz: File, out: File) {
        val entries = NpzParser.read(npz.readBytes())
        val nFft = scalar(entries, listOf("n_fft", "N_FFT"), DspConstants.N_FFT).toInt()
        val hop = scalar(entries, listOf("hop", "hop_length", "HOP"), DspConstants.HOP).toInt()
        val nMels = scalar(entries, listOf("n_mels", "N_MELS", "num_mels"), DspConstants.N_MELS).toInt()
        val nFreq = nFft / 2 + 1

        val fbEntry = NpzParser.findBy(entries, listOf("fb", "mel_fb", "filterbank")) { e ->
            if (e.floats != null) e else null
        } ?: return
        val win = NpzParser.findBy(entries, listOf("window", "win", "hann")) { e ->
            if (e.floats != null) e.floats else null
        } ?: return

        // The npz stores fb as (nFreq, nMels); the app wants (nMels, nFreq) so its
        // VocosFrontend can index mel[m * nFreq + f]. Transpose when needed.
        val fb = if (fbEntry.shape.size == 2 && fbEntry.shape[0] == nFreq && fbEntry.shape[1] == nMels) {
            val src = fbEntry.floats!!
            val t = FloatArray(nMels * nFreq)
            for (m in 0 until nMels) for (f in 0 until nFreq) t[m * nFreq + f] = src[f * nMels + m]
            t
        } else {
            fbEntry.floats!!
        }

        val buf = java.nio.ByteBuffer.allocate(12 + fb.size * 4 + win.size * 4)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        buf.putInt(nFft); buf.putInt(hop); buf.putInt(nMels)
        for (v in fb) buf.putFloat(v)
        for (v in win) buf.putFloat(v)
        out.writeBytes(buf.array())
    }

    private fun writeVocosHead(npz: File, out: File) {
        val entries = NpzParser.read(npz.readBytes())
        val nFft = scalar(entries, listOf("n_fft", "N_FFT"), DspConstants.N_FFT).toInt()
        val hop = scalar(entries, listOf("hop", "hop_length", "HOP"), DspConstants.HOP).toInt()
        val winLen = scalar(entries, listOf("win_length", "win_len"), DspConstants.N_FFT).toInt()

        val weight = NpzParser.findBy(entries, listOf("linear_weight", "weight")) { e ->
            if (e.floats != null && e.numElements == DspConstants.OUT_DIM * DspConstants.HIDDEN_DIM) e.floats else null
        } ?: return
        val bias = NpzParser.findBy(entries, listOf("linear_bias", "bias")) { e ->
            if (e.floats != null && e.numElements == DspConstants.OUT_DIM) e.floats else null
        } ?: return
        val window = NpzParser.findBy(entries, listOf("window", "win", "hann")) { e ->
            if (e.floats != null && e.numElements == winLen) e.floats else null
        } ?: return

        val buf = java.nio.ByteBuffer.allocate(12 + weight.size * 4 + bias.size * 4 + window.size * 4)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN)
        buf.putInt(nFft); buf.putInt(hop); buf.putInt(winLen)
        for (v in weight) buf.putFloat(v)
        for (v in bias) buf.putFloat(v)
        for (v in window) buf.putFloat(v)
        out.writeBytes(buf.array())
    }

    private fun scalar(entries: Map<String, NpzParser.NpzEntry>, cands: List<String>, default: Int): Long {
        val v = NpzParser.findBy(entries, cands) { e -> if (e.longs != null) e.scalarLong() else null }
        return v ?: default.toLong()
    }
}
