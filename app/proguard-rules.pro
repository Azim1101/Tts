# DhVaani TTS & MNN ProGuard rules
-keep class zone.dhvaani.tts.** { *; }
-keep interface zone.dhvaani.tts.** { *; }
-keepclasseswithmembernames class * { native <methods>; }

# Kotlin coroutines / lifecycle
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**
