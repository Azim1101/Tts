// dhvaani.h — DhVaani-0.5 on-device TTS engine (MNN backend)
// Pure C++17, no JNI. Same code compiles for Android (arm64-v8a) and desktop.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dhvaani {

struct Config {
    int   numThread     = 4;      // MNN threads
    int   precision     = 2;      // 0=Normal 1=High 2=Low(fp16 on arm64)
    int   numStep       = 8;      // flow-matching steps (quality vs speed)
    float guidanceScale = 1.0f;
    float speed         = 1.0f;   // >1 faster speech
    float tShift        = 0.5f;
    int   seed          = 666;
    bool  useGpu        = false;  // try OpenCL, falls back to CPU
    // Trade speed for RAM: no weight pre-packing, MNN Memory_Low, and fm_decoder
    // is loaded on demand and freed after each call.
    bool  lowMemory     = false;
};

struct Result {
    std::vector<float> samples;   // mono, [-1, 1]
    int   sampleRate = 24000;
    float rtf        = 0.0f;      // wall time / audio duration
    bool  ok         = false;
    std::string error;
};

// Progress callback: (stage, current, total). stage is one of
// "text_encoder", "fm_decoder", "vocoder". Return false to abort.
using ProgressFn = std::function<bool(const char* stage, int cur, int total)>;

class Engine {
public:
    Engine();
    ~Engine();

    // modelDir must contain:
    //   text_encoder_int8.mnn  fm_decoder_int8.mnn  vocoder_backbone.mnn
    //   mel_fb.bin  vocos_head.bin  tokens.txt
    bool init(const std::string& modelDir, const Config& cfg, std::string* err = nullptr);

    // Register the reference speaker. promptPcm must be mono float [-1,1].
    // Any sample rate is accepted; it is resampled to 24 kHz internally.
    bool setPrompt(const std::vector<float>& promptPcm, int sampleRate,
                   const std::string& promptText, std::string* err = nullptr);

    // Synthesize. setPrompt() must have been called first.
    Result synthesize(const std::string& text, ProgressFn onProgress = nullptr);

    // Runs one tiny forward pass so the first real call is not slow.
    void warmup();

    bool isReady() const;
    const Config& config() const;
    void setConfig(const Config& cfg);   // numStep/speed/guidance can change per call

    static std::string version();
    static bool writeWav(const std::string& path, const std::vector<float>& pcm, int sr);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

}  // namespace dhvaani
