// dhvaani.h — DhVaani-0.5 on-device TTS engine (MNN backend)
// Optimized for mobile devices: low memory footprint, zero-leak tensor reuse,
// sequential execution on demand.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dhvaani {

struct Config {
    int   numThread     = 2;      // Default 2 threads (balanced for mobile CPUs)
    int   precision     = 2;      // 0=Normal, 1=High, 2=Low (fp16 on arm64)
    int   numStep       = 8;      // Flow-matching steps
    float guidanceScale = 1.0f;
    float speed         = 1.0f;   // >1 faster speech
    float tShift        = 0.5f;
    int   seed          = 666;
    bool  useGpu        = false;  // OpenCL / Vulkan
    bool  lowMemory     = true;   // Default true: load models sequentially & free immediately
};

struct Result {
    std::vector<float> samples;   // Mono float samples, [-1, 1]
    int   sampleRate = 24000;
    float rtf        = 0.0f;      // Wall time / audio duration
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

    // Runs a minimal warmup pass.
    void warmup();

    bool isReady() const;
    const Config& config() const;
    void setConfig(const Config& cfg);

    static std::string version();
    static bool writeWav(const std::string& path, const std::vector<float>& pcm, int sr);

private:
    struct Impl;
    std::unique_ptr<Impl> d;
};

}  // namespace dhvaani
