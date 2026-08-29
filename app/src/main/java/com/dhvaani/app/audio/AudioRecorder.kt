package com.dhvaani.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Simple mono 16-bit microphone recorder at 44.1 kHz. Samples are streamed into
 * a growing list while recording and returned as normalized floats on stop.
 * The app subsequently resamples to 24 kHz for the model.
 */
class AudioRecorder(private val sampleRate: Int = 44100) {

    @Volatile
    var isRecording: Boolean = false
        private set

    private val active = AtomicBoolean(false)
    private val samples = ArrayList<Float>(sampleRate)
    private var thread: Thread? = null

    fun start() {
        if (active.get()) return
        samples.clear()
        active.set(true)
        isRecording = true

        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        val bufferSize = maxOf(minBuf, AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) * 2)

        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        if (record.state != AudioRecord.STATE_INITIALIZED) {
            active.set(false)
            isRecording = false
            record.release()
            return
        }

        thread = Thread {
            record.startRecording()
            val buf = ShortArray(bufferSize / 2)
            while (active.get()) {
                val n = record.read(buf, 0, buf.size)
                if (n > 0) {
                    for (i in 0 until n) {
                        samples.add(buf[i].toFloat() / 32768f)
                    }
                }
            }
            try {
                record.stop()
            } catch (_: Exception) {}
            record.release()
        }.apply { priority = Thread.MAX_PRIORITY; start() }
    }

    fun stop(): FloatArray {
        if (!active.getAndSet(false)) return FloatArray(0)
        isRecording = false
        thread?.join(2000)
        thread = null
        val out = samples.toFloatArray()
        samples.clear()
        return out
    }
}
