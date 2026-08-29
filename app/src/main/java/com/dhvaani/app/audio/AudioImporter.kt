package com.dhvaani.app.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class ImportedAudio(val samples: FloatArray, val sampleRate: Int)

/**
 * Decodes an imported audio file (mp3 / m4a / aac / ogg / flac / wav) to mono
 * 16-bit(float) PCM using MediaExtractor + MediaCodec. No platform media player
 * is used for playback so we never route audio through an external sink.
 */
class AudioImporter(private val context: Context) {

    fun decode(uri: Uri): ImportedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            var trackIndex = -1
            var mime = ""
            for (i in 0 until extractor.trackCount) {
                val fmt = extractor.getTrackFormat(i)
                val type = fmt.getString(MediaFormat.KEY_MIME) ?: ""
                if (type.startsWith("audio/")) {
                    trackIndex = i
                    mime = type
                    break
                }
            }
            check(trackIndex >= 0) { "No audio track found" }
            extractor.selectTrack(trackIndex)

            val format = extractor.getTrackFormat(trackIndex)
            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0)
            codec.start()

            val decoded = ArrayList<Float>(sampleRate * 8)
            val bufferInfo = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false

            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val inBuf = codec.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(inBuf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
                when {
                    outIdx >= 0 -> {
                        if (bufferInfo.size > 0) {
                            val outBuf = codec.getOutputBuffer(outIdx)!!
                            appendPcm(decoded, outBuf, bufferInfo, channelCount)
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* keep going */ }
                }
            }

            if (decoded.isEmpty()) throw IllegalStateException("Decoded no audio samples")
            return ImportedAudio(decoded.toFloatArray(), sampleRate)
        } finally {
            try { codec?.stop() } catch (_: Exception) {}
            try { codec?.release() } catch (_: Exception) {}
            extractor.release()
        }
    }

    private fun appendPcm(dst: MutableList<Float>, buf: ByteBuffer, info: MediaCodec.BufferInfo, channels: Int) {
        buf.position(info.offset)
        buf.limit(info.offset + info.size)
        val floats = FloatArray(info.size / 2)
        buf.order(ByteOrder.LITTLE_ENDIAN)
        for (i in floats.indices) {
            floats[i] = buf.short.toFloat() / 32768f
        }
        if (channels > 1) {
            val mono = FloatArray(floats.size / channels)
            for (i in mono.indices) {
                var acc = 0.0f
                for (c in 0 until channels) acc += floats[i * channels + c]
                mono[i] = acc / channels
            }
            for (v in mono) dst.add(v)
        } else {
            for (v in floats) dst.add(v)
        }
    }
}
