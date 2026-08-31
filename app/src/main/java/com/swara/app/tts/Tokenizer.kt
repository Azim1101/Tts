package com.swara.app.tts

/**
 * Character-level Indic tokenizer backed by the model's `tokens.txt`
 * (char <TAB> id, one single-character token per line).
 *
 * The tokenizer is intentionally simple and script-dependent: it maps each
 * code point to its integer id, silently DROPS characters that are not in the
 * vocabulary (Latin "Hinglish", digits, abbreviations, foreign words), and
 * errors out if nothing survives.
 */
class Tokenizer(charToId: Map<String, Int>) {

    private val map: Map<String, Int> = charToId

    /**
     * Ensure the text ends with a sentence terminator. If the last non-space
     * character is already one of the recognised punctuation marks it is left
     * alone, otherwise a period is appended.
     */
    fun addPunctuation(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return trimmed
        val last = trimmed.last()
        if (last in TERMINATORS) return trimmed
        return trimmed + "."
    }

    /** Map every surviving character to its id. Characters outside the vocab are skipped. */
    fun toIds(text: String): LongArray {
        val ids = ArrayList<Long>(text.length)
        for (ch in text) {
            val id = map[ch.toString()]
            if (id != null) ids.add(id.toLong())
        }
        return ids.toLongArray()
    }

    fun isVocabNonEmpty(): Boolean = map.isNotEmpty()

    companion object {
        // ; : , . ! ?               and their fullwidth variants
        private val TERMINATORS = setOf(';', ':', ',', '.', '!', '?', '；', '：', '，', '。', '！', '？', '।')

        /** Build a [Tokenizer] from the raw tokens.txt text (lines "char\tid"). */
        fun fromFileContent(content: String): Tokenizer {
            val map = HashMap<String, Int>()
            for (line in content.lineSequence()) {
                if (line.isBlank()) continue
                val idx = line.indexOf('\t')
                if (idx < 0) continue
                val ch = line.substring(0, idx)
                val idStr = line.substring(idx + 1).trim()
                val id = idStr.toIntOrNull() ?: continue
                if (ch.isNotEmpty()) map[ch] = id
            }
            return Tokenizer(map)
        }
    }
}
