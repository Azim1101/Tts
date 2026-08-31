package com.swara.app.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Thin wrapper around [AudioTrack] to stream short mono float buffers.
 * One instance holds at most one active track; [play] replaces any previous one.
 */
class AudioPlayer {

    @Volatile
    var isPlaying: Boolean = false
        private set

    private var track: AudioTrack? = null
    private val stopFlag = AtomicBoolean(false)

    fun play(samples: FloatArray, sampleRate: Int) {
        stop()
        val minBuf = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val bufferSize = maxOf(minBuf, sampleRate)
        val t = AudioTrack.Builder()
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
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(bufferSize)
            .build()

        if (t.state != AudioTrack.STATE_INITIALIZED) {
            t.release()
            return
        }

        track = t
        stopFlag.set(false)
        isPlaying = true
        t.play()

        val chunk = 4096
        var i = 0
        Thread {
            try {
                while (i < samples.size && !stopFlag.get() && t.playState == AudioTrack.PLAYSTATE_PLAYING) {
                    val end = minOf(i + chunk, samples.size)
                    val n = t.write(samples, i, end - i, AudioTrack.WRITE_BLOCKING)
                    if (n <= 0) break
                    i += n
                }
            } finally {
                if (track === t) {
                    try { t.stop() } catch (_: Exception) {}
                    try { t.release() } catch (_: Exception) {}
                    track = null
                    isPlaying = false
                }
            }
        }.apply { priority = Thread.MAX_PRIORITY; start() }
    }

    fun stop() {
        stopFlag.set(true)
        val t = track ?: return
        try { t.pause() } catch (_: Exception) {}
        try { t.stop() } catch (_: Exception) {}
        try { t.release() } catch (_: Exception) {}
        if (track === t) {
            track = null
            isPlaying = false
        }
    }
}
