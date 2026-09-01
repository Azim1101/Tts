// dhvaani.cpp — DhVaani-0.5 on-device TTS engine, MNN Express backend.
#include "dhvaani.h"

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>

using namespace MNN::Express;

namespace dhvaani {

// ------------------------------------------------------------------ constants
static constexpr int   kSR        = 24000;
static constexpr int   kNFFT      = 1024;
static constexpr int   kHop       = 256;
static constexpr int   kNMels     = 100;
static constexpr float kFeatScale = 0.1f;
static constexpr float kTargetRms = 0.1f;

// ------------------------------------------------------------------ small FFT
// Iterative radix-2 complex FFT. kNFFT is 1024 so this is always valid.
static void fftRadix2(std::vector<std::complex<float>>& a, bool inverse) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = (inverse ? 2.0f : -2.0f) * float(M_PI) / float(len);
        const std::complex<float> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<float> u = a[i + k];
                const std::complex<float> v = a[i + k + len / 2] * w;
                a[i + k]             = u + v;
                a[i + k + len / 2]   = u - v;
                w *= wl;
            }
        }
    }
    if (inverse) {
        for (auto& x : a) x /= float(n);
    }
}

// ------------------------------------------------------------------ resampler
// Windowed-sinc rational resampler (quality is plenty for a reference prompt).
static std::vector<float> resampleTo(const std::vector<float>& in, int srIn, int srOut) {
    if (srIn == srOut || in.empty()) return in;
    const double ratio  = double(srOut) / double(srIn);
    const size_t outN   = size_t(std::llround(double(in.size()) * ratio));
    const double cutoff = 0.5 * std::min(1.0, ratio);
    const int    taps   = 32;
    std::vector<float> out(outN, 0.0f);
    for (size_t i = 0; i < outN; ++i) {
        const double center = double(i) / ratio;
        const long   base   = long(std::floor(center));
        double acc = 0.0, wsum = 0.0;
        for (int k = -taps; k <= taps; ++k) {
            const long idx = base + k;
            if (idx < 0 || idx >= long(in.size())) continue;
            const double x = (double(idx) - center);
            const double s = (std::fabs(x) < 1e-9)
                                 ? 2.0 * cutoff
                                 : std::sin(2.0 * M_PI * cutoff * x) / (M_PI * x);
            // Blackman window
            const double t = (x + taps) / (2.0 * taps);
            const double w = 0.42 - 0.5 * std::cos(2 * M_PI * t) + 0.08 * std::cos(4 * M_PI * t);
            acc  += in[size_t(idx)] * s * w;
            wsum += s * w;
        }
        out[i] = float(wsum > 1e-9 ? acc / wsum : acc);
    }
    return out;
}

// ------------------------------------------------------------------ asset I/O
struct MelAssets {
    int n_fft = 0, hop = 0, n_mels = 0, n_freqs = 0;
    std::vector<float> fb;      // [n_freqs * n_mels], row-major
    std::vector<float> window;  // [n_fft]
};

struct VocosHead {
    int out_dim = 0, in_dim = 0, n_fft = 0, hop = 0, win_length = 0;
    std::vector<float> W;       // [out_dim * in_dim]
    std::vector<float> b;       // [out_dim]
    std::vector<float> window;  // [win_length]
};

static bool readAll(const std::string& p, std::vector<char>& buf) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    buf.resize(size_t(n));
    return bool(f.read(buf.data(), n));
}

static bool loadMel(const std::string& path, MelAssets& m, std::string* err) {
    std::vector<char> buf;
    if (!readAll(path, buf)) { if (err) *err = "cannot read " + path; return false; }
    if (buf.size() < 28 || std::memcmp(buf.data(), "DVMF", 4) != 0) {
        if (err) *err = "bad magic in " + path; return false;
    }
    const int32_t* h = reinterpret_cast<const int32_t*>(buf.data() + 4);
    m.n_fft = h[1]; m.hop = h[2]; m.n_mels = h[3]; m.n_freqs = h[4];
    const int cols = h[5];
    const size_t nfb = size_t(m.n_freqs) * size_t(cols);
    const char*  p   = buf.data() + 28;
    const size_t need = (nfb + size_t(m.n_fft)) * sizeof(float);
    if (buf.size() < 28 + need) { if (err) *err = "truncated " + path; return false; }
    m.fb.resize(nfb);      std::memcpy(m.fb.data(), p, nfb * sizeof(float));
    p += nfb * sizeof(float);
    m.window.resize(m.n_fft); std::memcpy(m.window.data(), p, size_t(m.n_fft) * sizeof(float));
    return true;
}

static bool loadHead(const std::string& path, VocosHead& v, std::string* err) {
    std::vector<char> buf;
    if (!readAll(path, buf)) { if (err) *err = "cannot read " + path; return false; }
    if (buf.size() < 28 || std::memcmp(buf.data(), "DVVH", 4) != 0) {
        if (err) *err = "bad magic in " + path; return false;
    }
    const int32_t* h = reinterpret_cast<const int32_t*>(buf.data() + 4);
    v.out_dim = h[1]; v.in_dim = h[2]; v.n_fft = h[3]; v.hop = h[4]; v.win_length = h[5];
    const size_t nW = size_t(v.out_dim) * size_t(v.in_dim);
    const size_t need = (nW + size_t(v.out_dim) + size_t(v.win_length)) * sizeof(float);
    if (buf.size() < 28 + need) { if (err) *err = "truncated " + path; return false; }
    const char* p = buf.data() + 28;
    v.W.resize(nW);            std::memcpy(v.W.data(), p, nW * sizeof(float));           p += nW * sizeof(float);
    v.b.resize(v.out_dim);     std::memcpy(v.b.data(), p, size_t(v.out_dim) * sizeof(float)); p += size_t(v.out_dim) * sizeof(float);
    v.window.resize(v.win_length); std::memcpy(v.window.data(), p, size_t(v.win_length) * sizeof(float));
    return true;
}

// ------------------------------------------------------------------ tokenizer
// tokens.txt is "<token>\t<id>" per line; tokens are single UTF-8 characters.
class Tokenizer {
public:
    bool load(const std::string& path, std::string* err) {
        std::ifstream f(path);
        if (!f) { if (err) *err = "cannot read " + path; return false; }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.rfind('\t');
            if (tab == std::string::npos) continue;
            const std::string tok = line.substr(0, tab);
            const std::string ids = line.substr(tab + 1);
            if (tok.empty() || ids.empty()) continue;
            try { map_[tok] = std::stoi(ids); } catch (...) {}
        }
        return !map_.empty();
    }

    // Split UTF-8 into codepoint-sized strings and map each to an id.
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        size_t i = 0;
        while (i < text.size()) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            size_t len = 1;
            if      ((c & 0x80) == 0x00) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            if (i + len > text.size()) len = 1;
            const std::string ch = text.substr(i, len);
            const auto it = map_.find(ch);
            if (it != map_.end()) out.push_back(it->second);
            i += len;
        }
        return out;
    }

    size_t size() const { return map_.size(); }

private:
    std::unordered_map<std::string, int> map_;
};

static bool endsWithPunct(const std::string& s) {
    // Must match the reference Python implementation exactly:
    //   PUNCT = set(";:,.!?；：，。！？")
    // Note: the Devanagari danda "।" is deliberately NOT in this set, so a "."
    // is appended after it — that is what the model was exported/validated with.
    static const char* kP[] = {";", ":", ",", ".", "!", "?",
                               "\xEF\xBC\x9B", "\xEF\xBC\x9A", "\xEF\xBC\x8C",
                               "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F"};
    for (const char* p : kP) {
        const size_t n = std::strlen(p);
        if (s.size() >= n && s.compare(s.size() - n, n, p) == 0) return true;
    }
    return false;
}

static std::string addPunct(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\t')) s.pop_back();
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\n' || s[b] == '\t')) ++b;
    s = s.substr(b);
    if (s.empty()) return ".";
    if (!endsWithPunct(s)) s += ".";
    return s;
}

// ------------------------------------------------------------------ mel fbank
static std::vector<float> vocosFbank(const std::vector<float>& wav, const MelAssets& mel,
                                     int* outFrames) {
    const int pad = kNFFT / 2;
    std::vector<float> wp(wav.size() + size_t(2 * pad), 0.0f);
    std::copy(wav.begin(), wav.end(), wp.begin() + pad);
    if (wp.size() < size_t(kNFFT)) wp.resize(size_t(kNFFT), 0.0f);

    const int nFrames = 1 + int((wp.size() - kNFFT) / kHop);
    const int nFreqs  = mel.n_freqs;
    std::vector<float> logmel(size_t(nFrames) * size_t(kNMels));
    std::vector<std::complex<float>> buf(kNFFT);
    std::vector<float> mag(static_cast<size_t>(nFreqs), 0.0f);

    for (int t = 0; t < nFrames; ++t) {
        const float* src = wp.data() + size_t(t) * kHop;
        for (int i = 0; i < kNFFT; ++i) buf[i] = {src[i] * mel.window[i], 0.0f};
        fftRadix2(buf, false);
        for (int f = 0; f < nFreqs; ++f) mag[f] = std::abs(buf[f]);
        float* row = logmel.data() + size_t(t) * kNMels;
        for (int m = 0; m < kNMels; ++m) {
            float acc = 0.0f;
            const float* fbCol = mel.fb.data() + m;      // fb is [n_freqs][n_mels]
            for (int f = 0; f < nFreqs; ++f) acc += mag[f] * fbCol[size_t(f) * kNMels];
            row[m] = std::log(std::max(acc, 1e-7f));
        }
    }
    // Trim / edge-pad to the frame count the model expects.
    const int want = int((long(wav.size()) + kHop / 2) / kHop);
    std::vector<float> out(size_t(want) * kNMels);
    for (int t = 0; t < want; ++t) {
        const int src = std::min(t, nFrames - 1);
        std::copy_n(logmel.data() + size_t(src) * kNMels, kNMels,
                    out.data() + size_t(t) * kNMels);
    }
    *outFrames = want;
    return out;
}

// ------------------------------------------------------------------ ISTFT
static std::vector<float> istftSame(const std::vector<float>& magSpec,
                                    const std::vector<float>& phaseSpec, int nBins, int T,
                                    const VocosHead& vh) {
    const int nfft = vh.n_fft, hop = vh.hop, wl = vh.win_length;
    const int pad = (wl - hop) / 2;
    const int outSize = (T - 1) * hop + wl;
    std::vector<float> y(size_t(outSize), 0.0f), env(size_t(outSize), 0.0f);
    std::vector<std::complex<float>> buf(nfft);

    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < nfft; ++k) buf[k] = {0.0f, 0.0f};
        for (int f = 0; f < nBins; ++f) {
            const float m = magSpec[size_t(f) * T + t];
            const float p = phaseSpec[size_t(f) * T + t];
            buf[f] = {m * std::cos(p), m * std::sin(p)};
            if (f > 0 && f < nfft - f) buf[nfft - f] = std::conj(buf[f]);
        }
        fftRadix2(buf, true);
        const size_t off = size_t(t) * hop;
        for (int k = 0; k < wl; ++k) {
            const float w = vh.window[k];
            y[off + k]   += buf[k].real() * w;
            env[off + k] += w * w;
        }
    }
    std::vector<float> outv(size_t(outSize - 2 * pad));
    for (size_t i = 0; i < outv.size(); ++i)
        outv[i] = y[i + pad] / std::max(env[i + pad], 1e-11f);
    return outv;
}

// ------------------------------------------------------------------ Impl
struct Engine::Impl {
    Config cfg;
    std::string dir;
    bool ready = false;

    std::shared_ptr<Executor::RuntimeManager> rtmgr;
    Module::Config modCfg;
    std::shared_ptr<Module> mText, mFm, mVoc;

    MelAssets mel;
    VocosHead head;
    Tokenizer tok;

    // prompt state
    std::vector<float> promptFeat;   // [pl * kNMels] already * kFeatScale
    int   promptLen = 0;
    float promptRms = 1.0f;
    std::vector<int> promptTokens;

    std::shared_ptr<Module> loadModule(const std::string& file,
                                       const std::vector<std::string>& in,
                                       const std::vector<std::string>& out) {
        const std::string p = dir + "/" + file;
        return std::shared_ptr<Module>(
            Module::load(in, out, p.c_str(), rtmgr, &modCfg));
    }

    static VARP scalarF(float v) {
        auto x = _Input({}, NCHW, halide_type_of<float>());
        x->writeMap<float>()[0] = v;
        return x;
    }
    static VARP scalarI(int v) {
        auto x = _Input({}, NCHW, halide_type_of<int>());
        x->writeMap<int>()[0] = v;
        return x;
    }
    static VARP tensorF(const std::vector<int>& shape, const float* data, size_t n) {
        auto x = _Input(shape, NCHW, halide_type_of<float>());
        std::memcpy(x->writeMap<float>(), data, n * sizeof(float));
        return x;
    }
    static VARP tensorI(const std::vector<int>& shape, const std::vector<int>& data) {
        auto x = _Input(shape, NCHW, halide_type_of<int>());
        std::memcpy(x->writeMap<int>(), data.data(), data.size() * sizeof(int));
        return x;
    }
};

// ------------------------------------------------------------------ Engine
Engine::Engine() : d(new Impl) {}
Engine::~Engine() = default;
bool Engine::isReady() const { return d->ready; }
const Config& Engine::config() const { return d->cfg; }
void Engine::setConfig(const Config& c) { d->cfg = c; }
std::string Engine::version() { return "DhVaani-0.5 MNN engine 1.1.0"; }

bool Engine::init(const std::string& modelDir, const Config& cfg, std::string* err) {
    d->cfg = cfg;
    d->dir = modelDir;

    if (!loadMel(modelDir + "/mel_fb.bin", d->mel, err)) return false;
    if (!loadHead(modelDir + "/vocos_head.bin", d->head, err)) return false;
    if (!d->tok.load(modelDir + "/tokens.txt", err)) return false;

    MNN::ScheduleConfig sc;
    sc.type      = cfg.useGpu ? MNN_FORWARD_OPENCL : MNN_FORWARD_CPU;
    sc.numThread = std::max(1, cfg.numThread);
    MNN::BackendConfig bc;
    bc.precision = (cfg.precision == 0)   ? MNN::BackendConfig::Precision_Normal
                   : (cfg.precision == 1) ? MNN::BackendConfig::Precision_High
                                          : MNN::BackendConfig::Precision_Low;
    bc.power  = MNN::BackendConfig::Power_Normal;
    bc.memory = cfg.lowMemory ? MNN::BackendConfig::Memory_Low
                              : MNN::BackendConfig::Memory_Normal;
    sc.backendConfig = &bc;

    d->rtmgr.reset(Executor::RuntimeManager::createRuntimeManager(sc));
    if (!d->rtmgr) {
        if (err) *err = "createRuntimeManager failed";
        return false;
    }
    // Keep the backend fixed so dynamic shapes don't trigger re-scheduling.
    d->rtmgr->setMode(MNN::Interpreter::Session_Backend_Fix);

    d->modCfg.shapeMutable = true;   // sequence length varies per utterance
    // rearrange pre-packs weights: faster kernels, but it holds a second copy of
    // the packed weights, which costs a few hundred MB on fm_decoder. Off when
    // lowMemory is requested.
    d->modCfg.rearrange    = !cfg.lowMemory;

    d->mText = d->loadModule("text_encoder_int8.mnn",
                             {"tokens", "prompt_tokens", "prompt_features_len", "speed"},
                             {"text_condition"});
    d->mVoc  = d->loadModule("vocoder_backbone.mnn", {"mels"}, {"hidden"});
    if (!d->cfg.lowMemory) {
        d->mFm = d->loadModule("fm_decoder_int8.mnn",
                               {"t", "x", "text_condition", "speech_condition", "guidance_scale"},
                               {"v"});
    }
    if (!d->mText || !d->mVoc || (!d->cfg.lowMemory && !d->mFm)) {
        if (err) *err = "Module::load failed — check that the .mnn files exist in " + modelDir;
        return false;
    }
    d->ready = true;
    return true;
}

bool Engine::setPrompt(const std::vector<float>& pcmIn, int sampleRate,
                       const std::string& promptText, std::string* err) {
    if (!d->ready) { if (err) *err = "engine not initialised"; return false; }
    if (pcmIn.empty()) { if (err) *err = "empty prompt audio"; return false; }

    std::vector<float> wav = resampleTo(pcmIn, sampleRate, kSR);

    double acc = 0.0;
    for (float v : wav) acc += double(v) * double(v);
    const float rms = float(std::sqrt(acc / double(wav.size())));
    d->promptRms = rms;
    if (rms > 0.0f && rms < kTargetRms) {
        const float g = kTargetRms / rms;
        for (float& v : wav) v *= g;
    }

    int frames = 0;
    std::vector<float> feats = vocosFbank(wav, d->mel, &frames);
    d->promptFeat.resize(feats.size());
    for (size_t i = 0; i < feats.size(); ++i) d->promptFeat[i] = feats[i] * kFeatScale;
    d->promptLen = frames;

    d->promptTokens = d->tok.encode(addPunct(promptText));
    if (d->promptTokens.empty()) { if (err) *err = "prompt text produced no tokens"; return false; }
    return true;
}

void Engine::warmup() {
    if (!d->ready || d->promptTokens.empty()) return;
    Config saved = d->cfg;
    d->cfg.numStep = 1;
    synthesize("\xE0\xA4\x85");   // single Devanagari char
    d->cfg = saved;
}

Result Engine::synthesize(const std::string& text, ProgressFn onProgress) {
    Result r;
    if (!d->ready)              { r.error = "engine not initialised"; return r; }
    if (d->promptTokens.empty()){ r.error = "call setPrompt() first";  return r; }

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<int> tokens = d->tok.encode(addPunct(text));
    if (tokens.empty()) { r.error = "text produced no tokens (unsupported script?)"; return r; }

    // ---- 1. text encoder -------------------------------------------------
    if (onProgress && !onProgress("text_encoder", 0, 1)) { r.error = "aborted"; return r; }
    std::vector<VARP> teIn = {
        Impl::tensorI({1, int(tokens.size())}, tokens),
        Impl::tensorI({1, int(d->promptTokens.size())}, d->promptTokens),
        Impl::scalarI(d->promptLen),
        Impl::scalarF(d->cfg.speed),
    };
    auto teOut = d->mText->onForward(teIn);
    if (teOut.empty() || teOut[0] == nullptr) { r.error = "text_encoder forward failed"; return r; }
    auto tcInfo = teOut[0]->getInfo();
    if (!tcInfo || tcInfo->dim.size() != 3) { r.error = "text_encoder bad output rank"; return r; }
    const int T = tcInfo->dim[1], D = tcInfo->dim[2];
    std::vector<float> textCond(size_t(T) * size_t(D));
    std::memcpy(textCond.data(), teOut[0]->readMap<float>(), textCond.size() * sizeof(float));
    teOut.clear();
    if (onProgress && !onProgress("text_encoder", 1, 1)) { r.error = "aborted"; return r; }

    // ---- 2. flow-matching decoder ---------------------------------------
    std::mt19937 gen(uint32_t(d->cfg.seed));
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(size_t(T) * size_t(D));
    for (auto& v : x) v = gauss(gen);

    std::vector<float> speechCond(size_t(T) * size_t(D), 0.0f);
    const int k = std::min(d->promptLen, T);
    std::copy_n(d->promptFeat.data(), size_t(k) * size_t(D), speechCond.data());

    const int N = std::max(1, d->cfg.numStep);
    std::vector<float> ts(size_t(N) + 1);
    for (int i = 0; i <= N; ++i) {
        const float u = float(i) / float(N);
        ts[size_t(i)] = d->cfg.tShift * u / (1.0f + (d->cfg.tShift - 1.0f) * u);
    }

    if (d->cfg.lowMemory && !d->mFm) {
        d->mFm = d->loadModule("fm_decoder_int8.mnn",
                               {"t", "x", "text_condition", "speech_condition", "guidance_scale"},
                               {"v"});
        if (!d->mFm) { r.error = "fm_decoder load failed"; return r; }
    }

    const std::vector<int> shp = {1, T, D};
    auto vTextCond   = Impl::tensorF(shp, textCond.data(), textCond.size());
    auto vSpeechCond = Impl::tensorF(shp, speechCond.data(), speechCond.size());
    auto vGuidance   = Impl::scalarF(d->cfg.guidanceScale);

    for (int step = 0; step < N; ++step) {
        if (onProgress && !onProgress("fm_decoder", step, N)) { r.error = "aborted"; return r; }
        std::vector<VARP> in = {
            Impl::scalarF(ts[size_t(step)]),
            Impl::tensorF(shp, x.data(), x.size()),
            vTextCond, vSpeechCond, vGuidance,
        };
        auto out = d->mFm->onForward(in);
        if (out.empty() || out[0] == nullptr) { r.error = "fm_decoder forward failed"; return r; }
        const float* v  = out[0]->readMap<float>();
        const float  dt = ts[size_t(step) + 1] - ts[size_t(step)];
        for (size_t i = 0; i < x.size(); ++i) x[i] += v[i] * dt;
    }
    if (onProgress && !onProgress("fm_decoder", N, N)) { r.error = "aborted"; return r; }

    if (d->cfg.lowMemory) { d->mFm.reset(); }

    // ---- 3. vocoder ------------------------------------------------------
    if (onProgress && !onProgress("vocoder", 0, 1)) { r.error = "aborted"; return r; }
    const int Tp = T - k;
    if (Tp <= 0) { r.error = "prompt longer than generated sequence"; return r; }

    // mels: (1, D, Tp) = transpose of x[k:] / featScale
    std::vector<float> mels(size_t(D) * size_t(Tp));
    for (int t = 0; t < Tp; ++t)
        for (int c = 0; c < D; ++c)
            mels[size_t(c) * size_t(Tp) + size_t(t)] =
                x[size_t(k + t) * size_t(D) + size_t(c)] / kFeatScale;

    std::vector<VARP> vIn = { Impl::tensorF({1, D, Tp}, mels.data(), mels.size()) };
    auto vOut = d->mVoc->onForward(vIn);
    if (vOut.empty() || vOut[0] == nullptr) { r.error = "vocoder forward failed"; return r; }
    auto hInfo = vOut[0]->getInfo();
    if (!hInfo || hInfo->dim.size() != 3) { r.error = "vocoder bad output rank"; return r; }
    const int Th = hInfo->dim[1], H = hInfo->dim[2];
    const float* hid = vOut[0]->readMap<float>();

    // vocos head: y = hidden @ W^T + b   ->  (Th, out_dim)
    const int OD = d->head.out_dim;
    const int nBins = OD / 2;
    std::vector<float> magS(size_t(nBins) * size_t(Th));
    std::vector<float> phaS(size_t(nBins) * size_t(Th));
    const float logCap = std::log(1e2f);
    for (int t = 0; t < Th; ++t) {
        const float* hrow = hid + size_t(t) * size_t(H);
        for (int o = 0; o < OD; ++o) {
            const float* w = d->head.W.data() + size_t(o) * size_t(d->head.in_dim);
            float acc = d->head.b[size_t(o)];
            for (int i = 0; i < H; ++i) acc += hrow[i] * w[i];
            if (o < nBins) magS[size_t(o) * size_t(Th) + size_t(t)] =
                               std::exp(std::min(acc, logCap));
            else           phaS[size_t(o - nBins) * size_t(Th) + size_t(t)] = acc;
        }
    }
    vOut.clear();

    std::vector<float> audio = istftSame(magS, phaS, nBins, Th, d->head);
    if (d->promptRms < kTargetRms && d->promptRms > 0.0f) {
        const float g = d->promptRms / kTargetRms;
        for (float& v : audio) v *= g;
    }
    for (float& v : audio) v = std::max(-1.0f, std::min(1.0f, v));
    if (onProgress) onProgress("vocoder", 1, 1);

    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double dur  = double(audio.size()) / double(kSR);

    r.samples    = std::move(audio);
    r.sampleRate = kSR;
    r.rtf        = float(dur > 1e-6 ? wall / dur : 0.0);
    r.ok         = true;
    return r;
}

bool Engine::writeWav(const std::string& path, const std::vector<float>& pcm, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t n = uint32_t(pcm.size());
    const uint32_t dataBytes = n * 2;
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("RIFF", 4); u32(36 + dataBytes); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(1); u16(1);
    u32(uint32_t(sr)); u32(uint32_t(sr) * 2); u16(2); u16(16);
    f.write("data", 4); u32(dataBytes);
    for (float v : pcm) {
        const int s = int(std::lround(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f));
        u16(uint16_t(int16_t(s)));
    }
    return true;
}

}  // namespace dhvaani
