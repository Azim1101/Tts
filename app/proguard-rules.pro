# onnxruntime Android is shipped without reflection-heavy code, but keep the
# native JNI entry points and the public API surface so R8 does not strip them.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
-dontnote ai.onnxruntime.**

# Kotlin coroutines / lifecycle
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**
