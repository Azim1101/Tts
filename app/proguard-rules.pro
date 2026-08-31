# MNN Android runtime (Native + Java bindings). Keep the JNI entry points and
# the wrapper API so R8 does not strip them in release builds.
-keep class com.taobao.android.mnn.** { *; }
-keep class com.alibaba.mnn.** { *; }
-keep class com.alibaba.android.mnn.** { *; }
-dontwarn com.taobao.android.mnn.**
-dontwarn com.alibaba.mnn.**
-dontwarn com.alibaba.android.mnn.**

# Kotlin coroutines / lifecycle
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**
