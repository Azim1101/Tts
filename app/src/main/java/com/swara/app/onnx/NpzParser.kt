package com.swara.app.onnx

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal reader for NumPy `.npz` (a zip of `.npy` arrays) so the app can turn the
 * `mel_fb.npz` / `vocos_head.npz` feature files into the same little-endian `.bin`
 * format that `scripts/make_assets.py` produces. No Python/NumPy is used at runtime.
 *
 * Only the array types needed here are supported: float32/float64 (for the
 * filterbank/window/weight/bias) and int32/int64 scalars (n_fft/hop/n_mels etc.).
 *
 * Layout of a `.npy` file:
 *   magic \x93NUMPY (6 bytes) | major(1) minor(1) | headerLen(2 or 4 LE) |
 *   descr/shape dict (ASCII) | data (padded so the header section is 64-aligned)
 */
object NpzParser {

    /** One array read from the npz. Exactly one of [floats]/[longs] is populated. */
    class NpzEntry(
        val name: String,
        val shape: IntArray,
        val floats: FloatArray?,
        val longs: LongArray?
    ) {
        val numElements: Int = shape.fold(1) { acc, d -> acc * d }

        fun scalarLong(): Long = longs?.getOrElse(0) { 0L } ?: 0L

        /** True when the entry matches the given flattened length and is float. */
        fun isFloatLen(len: Int): Boolean = floats != null && numElements == len
    }

    private val descrRegex = Regex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]")
    private val shapeRegex = Regex("['\"]shape['\"]\\s*:\\s*(\\([^)]*\\))")

    fun read(bytes: ByteArray): Map<String, NpzEntry> {
        val out = HashMap<String, NpzEntry>()
        val zip = java.util.zip.ZipInputStream(bytes.inputStream())
        var entry = zip.nextEntry
        while (entry != null) {
            val name = entry.name
            if (name.endsWith(".npy")) {
                val data = zip.readBytes()
                val key = name.removeSuffix(".npy")
                out[key] = parseNpy(key, data)
            }
            zip.closeEntry()
            entry = zip.nextEntry
        }
        zip.close()
        return out
    }

    private fun parseNpy(name: String, data: ByteArray): NpzEntry {
        require(data.size >= 10 && data[0] == 0x93.toByte() && data[1].toInt() == 'N'.code &&
            data[2].toInt() == 'U'.code && data[3].toInt() == 'M'.code &&
            data[4].toInt() == 'P'.code && data[5].toInt() == 'Y'.code) { "Bad npy magic" }

        val major = data[6].toInt() and 0xFF
        val minor = data[7].toInt() and 0xFF
        val (hlen, headerStart) = when (major) {
            1 -> {
                val h = ((data[9].toInt() and 0xFF) shl 8) or (data[8].toInt() and 0xFF)
                h to 10
            }
            2 -> {
                val h = ((data[11].toInt() and 0xFF) shl 24) or ((data[10].toInt() and 0xFF) shl 16) or
                    ((data[9].toInt() and 0xFF) shl 8) or (data[8].toInt() and 0xFF)
                h to 12
            }
            else -> error("Unsupported npy version $major")
        }

        val header = String(data, headerStart, hlen, Charsets.US_ASCII)
        val descr = descrRegex.find(header)?.groupValues?.get(1) ?: "<f4"
        val shapeStr = shapeRegex.find(header)?.groupValues?.get(1) ?: "()"
        val shape = parseShape(shapeStr)

        val typeChar = descr.getOrNull(1) ?: 'f'
        val sizeChar = Character.getNumericValue(descr.getOrNull(2) ?: '4').let { if (it < 0) 4 else it }
        val itemsize = sizeChar
        val isFloat = typeChar == 'f' || typeChar == 'd'

        val nelem = shape.fold(1) { acc, d -> acc * d }
        var off = headerStart + hlen
        if (off % 64 != 0) off += 64 - (off % 64)

        if (isFloat) {
            val floats = FloatArray(nelem)
            if (nelem > 0) {
                ByteBuffer.wrap(data, off, nelem * itemsize).order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer().get(floats)
            }
            return NpzEntry(name, shape, floats, null)
        } else {
            val longs = LongArray(nelem)
            if (nelem > 0) {
                val b = ByteBuffer.wrap(data, off, nelem * itemsize).order(ByteOrder.LITTLE_ENDIAN)
                if (itemsize == 8) b.asLongBuffer().get(longs)
                else {
                    for (i in 0 until nelem) longs[i] = b.int.toLong()
                }
            }
            return NpzEntry(name, shape, null, longs)
        }
    }

    private fun parseShape(s: String): IntArray {
        val inner = s.removePrefix("(").removeSuffix(")")
        val parts = inner.split(",").map { it.trim() }.filter { it.isNotEmpty() }
        return parts.map { it.toInt() }.toIntArray()
    }

    /** Look an entry up by any of the candidate names (the npz key naming is not guaranteed). */
    fun <T> findBy(entries: Map<String, NpzEntry>, cands: List<String>, pred: (NpzEntry) -> T?): T? {
        for (c in cands) {
            val e = entries[c] ?: continue
            val r = pred(e) ?: continue
            return r
        }
        return null
    }
}
