package com.swara.app.audio

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Writes 16-bit mono PCM as a RIFF/WAVE file (used for temporary stores and preview). */
object WavWriter {

    @Throws(IOException::class)
    fun write(file: File, samples: FloatArray, sampleRate: Int) {
        val dataBytes = samples.size * 2
        val total = 44 + dataBytes
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(total - 8)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)                          // PCM
        header.putShort(1)                          // mono
        header.putInt(sampleRate)
        header.putInt(sampleRate * 2)               // byte rate
        header.putShort(2)                          // block align
        header.putShort(16)                         // bits per sample
        header.put("data".toByteArray())
        header.putInt(dataBytes)

        FileOutputStream(file).use { out ->
            out.write(header.array())
            val pcm = ByteBuffer.allocate(dataBytes).order(ByteOrder.LITTLE_ENDIAN)
            for (s in samples) {
                val v = (s.coerceIn(-1f, 1f) * 32767f).toInt()
                pcm.putShort(v.toShort())
            }
            out.write(pcm.array())
        }
    }
}
