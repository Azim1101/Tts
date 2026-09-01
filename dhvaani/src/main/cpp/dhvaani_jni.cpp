// dhvaani_jni.cpp — JNI bridge for zone.dhvaani.tts.DhVaani
#include <jni.h>

#include <android/log.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "dhvaani.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "DhVaani", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "DhVaani", __VA_ARGS__)

namespace {

struct Handle {
    dhvaani::Engine engine;
    std::string lastError;
};

inline Handle* toHandle(jlong p) { return reinterpret_cast<Handle*>(p); }

std::string jstr(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

// Calls back into Kotlin: boolean onProgress(String stage, int cur, int total)
struct ProgressBridge {
    JNIEnv*  env    = nullptr;
    jobject  cb     = nullptr;
    jmethodID method = nullptr;

    bool valid() const { return env && cb && method; }

    bool call(const char* stage, int cur, int total) {
        if (!valid()) return true;
        jstring js = env->NewStringUTF(stage);
        jboolean keep = env->CallBooleanMethod(cb, method, js, jint(cur), jint(total));
        env->DeleteLocalRef(js);
        if (env->ExceptionCheck()) { env->ExceptionClear(); return false; }
        return keep == JNI_TRUE;
    }
};

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeCreate(JNIEnv* env, jobject,
                                           jstring modelDir, jint numThread, jint precision,
                                           jboolean useGpu, jboolean lowMemory) {
    auto* h = new Handle();
    dhvaani::Config cfg;
    cfg.numThread = numThread;
    cfg.precision = precision;
    cfg.useGpu    = (useGpu == JNI_TRUE);
    cfg.lowMemory = (lowMemory == JNI_TRUE);

    std::string err;
    const std::string dir = jstr(env, modelDir);
    if (!h->engine.init(dir, cfg, &err)) {
        LOGE("init failed: %s", err.c_str());
        h->lastError = err;
        // Keep the handle alive so Kotlin can read the error message.
        return reinterpret_cast<jlong>(h);
    }
    LOGI("engine ready (%s), models in %s", dhvaani::Engine::version().c_str(), dir.c_str());
    return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeDestroy(JNIEnv*, jobject, jlong p) {
    delete toHandle(p);
}

JNIEXPORT jboolean JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeIsReady(JNIEnv*, jobject, jlong p) {
    auto* h = toHandle(p);
    return (h && h->engine.isReady()) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeLastError(JNIEnv* env, jobject, jlong p) {
    auto* h = toHandle(p);
    return env->NewStringUTF(h ? h->lastError.c_str() : "null handle");
}

JNIEXPORT jboolean JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeSetPrompt(JNIEnv* env, jobject, jlong p,
                                              jfloatArray pcm, jint sampleRate, jstring text) {
    auto* h = toHandle(p);
    if (!h || !h->engine.isReady()) return JNI_FALSE;

    const jsize n = env->GetArrayLength(pcm);
    std::vector<float> buf(static_cast<size_t>(n), 0.0f);
    env->GetFloatArrayRegion(pcm, 0, n, buf.data());

    std::string err;
    if (!h->engine.setPrompt(buf, sampleRate, jstr(env, text), &err)) {
        h->lastError = err;
        LOGE("setPrompt failed: %s", err.c_str());
        return JNI_FALSE;
    }
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeConfigure(JNIEnv*, jobject, jlong p,
                                              jint numStep, jfloat guidance, jfloat speed,
                                              jint seed) {
    auto* h = toHandle(p);
    if (!h) return;
    dhvaani::Config c = h->engine.config();
    c.numStep       = numStep;
    c.guidanceScale = guidance;
    c.speed         = speed;
    c.seed          = seed;
    h->engine.setConfig(c);
}

JNIEXPORT void JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeWarmup(JNIEnv*, jobject, jlong p) {
    auto* h = toHandle(p);
    if (h) h->engine.warmup();
}

JNIEXPORT jfloatArray JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeSynthesize(JNIEnv* env, jobject, jlong p,
                                               jstring text, jobject progressCb) {
    auto* h = toHandle(p);
    if (!h || !h->engine.isReady()) return nullptr;

    ProgressBridge pb;
    if (progressCb != nullptr) {
        jclass cls = env->GetObjectClass(progressCb);
        pb.env    = env;
        pb.cb     = progressCb;
        pb.method = env->GetMethodID(cls, "onProgress", "(Ljava/lang/String;II)Z");
        env->DeleteLocalRef(cls);
    }

    dhvaani::ProgressFn fn = nullptr;
    if (pb.valid()) {
        fn = [&pb](const char* s, int c, int t) { return pb.call(s, c, t); };
    }

    auto r = h->engine.synthesize(jstr(env, text), fn);
    if (!r.ok) {
        h->lastError = r.error;
        LOGE("synthesize failed: %s", r.error.c_str());
        return nullptr;
    }
    LOGI("synthesized %.2fs audio, RTF=%.2f", r.samples.size() / float(r.sampleRate), r.rtf);
    h->lastError = "rtf=" + std::to_string(r.rtf);

    jfloatArray out = env->NewFloatArray(jsize(r.samples.size()));
    if (out == nullptr) return nullptr;
    env->SetFloatArrayRegion(out, 0, jsize(r.samples.size()), r.samples.data());
    return out;
}

JNIEXPORT jstring JNICALL
Java_zone_dhvaani_tts_DhVaani_nativeVersion(JNIEnv* env, jclass) {
    return env->NewStringUTF(dhvaani::Engine::version().c_str());
}

}  // extern "C"
