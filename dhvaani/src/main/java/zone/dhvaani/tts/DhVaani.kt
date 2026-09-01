package zone.dhvaani.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * DhVaani-0.5 on-device text-to-speech (MNN backend).
 *
 * Basic usage:
 * ```
 * val tts = DhVaani.fromAssets(context)          // copies models out of assets on first run
 * tts.setPrompt(promptPcm, 24000, "reference text")
 * val audio = tts.synthesize("नमस्ते दुनिया")     // FloatArray, 24 kHz mono
 * tts.play(audio)
 * tts.close()
 * ```
 *
 * Not thread-safe: call from a single background thread (never the main thread —
 * synthesis takes seconds).
 */
class DhVaani private constructor(
    modelDir: String,
    threads: Int,
    precision: Precision,
    useGpu: Boolean,
    lowMemory: Boolean,
) : Closeable {

    enum class Precision(val value: Int) {
        /** float32 everywhere. Slowest, most accurate. */
        NORMAL(0),
        /** float32 with high-precision kernels. */
        HIGH(1),
        /** fp16 on arm64 — ~1.6x faster, recommended. */
        LOW(2),
    }

    /** Return false from [onProgress] to abort synthesis. */
    fun interface ProgressListener {
        fun onProgress(stage: String, current: Int, total: Int): Boolean
    }

    private var handle: Long = nativeCreate(
        modelDir, threads, precision.value, useGpu, lowMemory
    )

    /** True when all three models loaded successfully. */
    val isReady: Boolean
        get() = handle != 0L && nativeIsReady(handle)

    /** Human-readable reason the last call failed. */
    fun lastError(): String = if (handle == 0L) "closed" else nativeLastError(handle)

    /**
     * Register the reference speaker. This defines the output voice.
     *
     * @param pcm mono samples in [-1, 1]
     * @param sampleRate any rate; resampled to 24 kHz internally
     * @param text exactly what is spoken in [pcm] — accuracy matters a lot for quality
     */
    fun setPrompt(pcm: FloatArray, sampleRate: Int, text: String): Boolean {
        check(handle != 0L) { "DhVaani is closed" }
        return nativeSetPrompt(handle, pcm, sampleRate, text)
    }

    /**
     * Tune generation. Call any time before [synthesize].
     *
     * @param numStep flow-matching steps. 4 = fast/rough, 8 = balanced (default), 16 = best.
     * @param guidanceScale classifier-free guidance; 1.0 is the validated default.
     * @param speed >1.0 speaks faster, <1.0 slower.
     * @param seed change for a different random realisation of the same text.
     */
    @JvmOverloads
    fun configure(
        numStep: Int = 8,
        guidanceScale: Float = 1.0f,
        speed: Float = 1.0f,
        seed: Int = 666,
    ) {
        check(handle != 0L) { "DhVaani is closed" }
        nativeConfigure(handle, numStep, guidanceScale, speed, seed)
    }

    /** Runs one throwaway pass so the first real call is not slow. Takes a few seconds. */
    fun warmup() {
        if (handle != 0L) nativeWarmup(handle)
    }

    /**
     * Synthesize [text]. Blocking; run off the main thread.
     * @return 24 kHz mono float samples, or null on failure (see [lastError]).
     */
    @JvmOverloads
    fun synthesize(text: String, listener: ProgressListener? = null): FloatArray? {
        check(handle != 0L) { "DhVaani is closed" }
        return nativeSynthesize(handle, text, listener)
    }

    /** Blocking playback through AudioTrack. */
    fun play(pcm: FloatArray, sampleRate: Int = SAMPLE_RATE) {
        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT
        ).coerceAtLeast(4096)
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBuf)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        try {
            track.play()
            var off = 0
            while (off < pcm.size) {
                val n = minOf(4096, pcm.size - off)
                val written = track.write(pcm, off, n, AudioTrack.WRITE_BLOCKING)
                if (written <= 0) break
                off += written
            }
            // Let the tail drain instead of clipping the last word.
            Thread.sleep(120)
        } finally {
            runCatching { track.stop() }
            track.release()
        }
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    // ---------------------------------------------------------------- native
    private external fun nativeCreate(
        modelDir: String, numThread: Int, precision: Int, useGpu: Boolean, lowMemory: Boolean
    ): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeIsReady(handle: Long): Boolean
    private external fun nativeLastError(handle: Long): String
    private external fun nativeSetPrompt(
        handle: Long, pcm: FloatArray, sampleRate: Int, text: String
    ): Boolean
    private external fun nativeConfigure(
        handle: Long, numStep: Int, guidance: Float, speed: Float, seed: Int
    )
    private external fun nativeWarmup(handle: Long)
    private external fun nativeSynthesize(
        handle: Long, text: String, listener: ProgressListener?
    ): FloatArray?

    companion object {
        const val SAMPLE_RATE = 24000
        private const val TAG = "DhVaani"

        /** Files that must be present in the model directory. */
        val REQUIRED_FILES = listOf(
            "text_encoder_int8.mnn",
            "fm_decoder_int8.mnn",
            "vocoder_backbone.mnn",
            "mel_fb.bin",
            "vocos_head.bin",
            "tokens.txt",
        )

        init {
            // libMNN_Express is required: these models contain subgraphs and can
            // only run through MNN::Express::Module.
            System.loadLibrary("MNN")
            runCatching { System.loadLibrary("MNN_Express") }
                .onFailure { Log.w(TAG, "libMNN_Express.so not found — ok if MNN was built with MNN_SEP_BUILD=OFF") }
            System.loadLibrary("dhvaani")
        }

        @JvmStatic
        external fun nativeVersion(): String

        /**
         * Open models from a directory you manage yourself (e.g. downloaded at runtime).
         */
        @JvmStatic
        @JvmOverloads
        fun fromDirectory(
            modelDir: File,
            threads: Int = defaultThreads(),
            precision: Precision = Precision.LOW,
            useGpu: Boolean = false,
            lowMemory: Boolean = false,
        ): DhVaani {
            val missing = REQUIRED_FILES.filterNot { File(modelDir, it).isFile }
            require(missing.isEmpty()) { "missing model files in $modelDir: $missing" }
            return DhVaani(modelDir.absolutePath, threads, precision, useGpu, lowMemory)
        }

        /**
         * Copy the models out of `assets/dhvaani/` into internal storage on first
         * launch, then open them.
         *
         * Note: mmap needs real files, so assets must be extracted. Make sure the
         * .mnn files are stored uncompressed (see `androidResources.noCompress`
         * in the module's build.gradle) or extraction will be slow.
         */
        @JvmStatic
        @JvmOverloads
        fun fromAssets(
            context: Context,
            assetSubdir: String = "dhvaani",
            threads: Int = defaultThreads(),
            precision: Precision = Precision.LOW,
            useGpu: Boolean = false,
            lowMemory: Boolean = false,
        ): DhVaani {
            val dst = File(context.filesDir, assetSubdir).apply { mkdirs() }
            for (name in REQUIRED_FILES) {
                val out = File(dst, name)
                val src = "$assetSubdir/$name"
                val expected = runCatching {
                    context.assets.openFd(src).use { it.length }
                }.getOrDefault(-1L)
                if (out.isFile && (expected <= 0 || out.length() == expected)) continue
                context.assets.open(src).use { input ->
                    FileOutputStream(out).use { output -> input.copyTo(output, 1 shl 20) }
                }
                Log.i(TAG, "extracted $name (${out.length()} bytes)")
            }
            return fromDirectory(dst, threads, precision, useGpu, lowMemory)
        }

        /** Leave one core free for the UI; cap at 4 (more rarely helps). */
        @JvmStatic
        fun defaultThreads(): Int =
            (Runtime.getRuntime().availableProcessors() - 1).coerceIn(1, 4)

        /** Read a 16-bit PCM WAV into mono floats. Returns samples to `sampleRate`. */
        @JvmStatic
        fun readWav(file: File): Pair<FloatArray, Int> {
            RandomAccessFile(file, "r").use { raf ->
                val header = ByteArray(12)
                raf.readFully(header)
                require(String(header, 0, 4) == "RIFF" && String(header, 8, 4) == "WAVE") {
                    "not a RIFF/WAVE file: $file"
                }
                var channels = 1
                var rate = SAMPLE_RATE
                var bits = 16
                var dataOffset = -1L
                var dataSize = 0
                val chunk = ByteArray(8)
                while (raf.filePointer < raf.length() - 8) {
                    raf.readFully(chunk)
                    val id = String(chunk, 0, 4)
                    val size = ByteBuffer.wrap(chunk, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
                    when (id) {
                        "fmt " -> {
                            val fmt = ByteArray(size)
                            raf.readFully(fmt)
                            val bb = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN)
                            bb.short                       // audio format
                            channels = bb.short.toInt()
                            rate = bb.int
                            bb.int; bb.short               // byte rate, block align
                            bits = bb.short.toInt()
                        }
                        "data" -> {
                            dataOffset = raf.filePointer
                            dataSize = size
                            raf.seek(raf.filePointer + size)
                        }
                        else -> raf.seek(raf.filePointer + size + (size and 1))
                    }
                }
                require(dataOffset >= 0) { "no data chunk in $file" }
                require(bits == 16) { "only 16-bit PCM supported, got $bits-bit" }
                raf.seek(dataOffset)
                val bytes = ByteArray(dataSize)
                raf.readFully(bytes)
                val sb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                val frames = dataSize / 2 / channels
                val out = FloatArray(frames)
                for (i in 0 until frames) {
                    var acc = 0f
                    for (c in 0 until channels) acc += sb.get(i * channels + c) / 32768f
                    out[i] = acc / channels
                }
                return out to rate
            }
        }

        /** Write mono floats as a 16-bit PCM WAV. */
        @JvmStatic
        @JvmOverloads
        fun writeWav(file: File, pcm: FloatArray, sampleRate: Int = SAMPLE_RATE) {
            val dataBytes = pcm.size * 2
            val bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
            bb.put("RIFF".toByteArray()); bb.putInt(36 + dataBytes); bb.put("WAVE".toByteArray())
            bb.put("fmt ".toByteArray()); bb.putInt(16)
            bb.putShort(1); bb.putShort(1)
            bb.putInt(sampleRate); bb.putInt(sampleRate * 2)
            bb.putShort(2); bb.putShort(16)
            bb.put("data".toByteArray()); bb.putInt(dataBytes)
            for (v in pcm) {
                bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
            }
            FileOutputStream(file).use { it.write(bb.array()) }
        }
    }
}
