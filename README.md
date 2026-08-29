# DhVaani — Android Hindi Voice Cloner (Fully Offline)

An on-device, zero-shot **voice cloning** + **text-to-speech** app for Android,
in a Hindi (English-fallback) UI. You record or import a reference voice ~3–8 s,
type what it says, then type any Indic text — the app speaks it in the reference
speaker's voice at 24 kHz. **No server, no audio ever leaves the device.**

Built on a torch-free ONNX export of `ARTPARK-IISc/DhVaani-0.5` (a ZipVoice
fine-tune, 27 Indic languages, zero-shot cloning) plus the `charactr/vocos-mel-24khz`
vocoder. **Apache-2.0.**

---

## What's inside

| Path | Purpose |
|---|---|
| `app/` | Android app (Kotlin, Material3, ViewBinding) |
| `app/src/main/java/com/dhvaani/app/dsp/` | Pure-Kotlin DSP: windowed-sinc resampler, log-mel frontend, ISTFT vocoder, RMS norm |
| `app/src/main/java/com/dhvaani/app/onnx/` | ONNX session wrapper + model manager |
| `app/src/main/java/com/dhvaani/app/tts/` | Tokenizer + synthesis pipeline (flow-matching Euler sampler) |
| `app/src/main/java/com/dhvaani/app/audio/` | AudioRecord recorder, MediaCodec importer, AudioTrack player, WAV/MediaStore saver |
| `scripts/download_models.sh` | Fetch ONNX graphs + features, convert `.npz` → `.bin` |
| `scripts/make_assets.py` | Numpy → little-endian `.bin` converter |
| `scripts/build_colab.ipynb` | Self-contained Colab build (JDK 17 + SDK 34 + Gradle 8.7) |

## Model assets (downloaded, not committed)

The 3 int8 ONNX graphs and small feature files come from
`https://huggingface.co/Bbkblo/DhVaani-0.5-ONNX`:

```
text_encoder_int8.onnx   (~6 MB)   chars + duration expand     -> text_condition [1,T,100]
fm_decoder_int8.onnx     (~119 MB) flow-matching velocity (CFG) -> v [1,T,100]
vocoder_backbone.onnx    (~50 MB)  Vocos ConvNeXt backbone      -> hidden [1,T,512]
vocos_head.npz                        linear head + window
mel_fb.npz                            HTK mel filterbank + window
tokens.txt                            char->id vocab (1058)
```

If the int8 files are absent the app automatically falls back to the fp32
`text_encoder.onnx` / `fm_decoder.onnx` variants (`ModelManager`).

## Backend note — why the 3 ONNX graphs are separate

`text_encoder` and `fm_decoder` are single nodes so onnxruntime can run them on
CPU without a full PyTorch runtime. The Vocos `vocoder_backbone.onnx` only
encodes the ConvNeXt; the cheap linear head and the overlap-add ISTFT are done in
pure Kotlin (no numpy / torch in the app).

---

## Build

### Option A — Local (Android Studio or CLI)

Requirements: **JDK 17**, **Android SDK 34** (`platforms;android-34`,
`build-tools;34.0.0`), **Gradle 8.7**, `python3` + `numpy` (for asset conversion),
and `curl`.

```bash
# 1. Download the ONNX graphs + features, convert npz -> bin into app/src/main/assets
bash scripts/download_models.sh

# 2. Build (thin gradlew delegates to your Gradle 8.7)
./gradlew assembleDebug
# or with a globally installed Gradle 8.7:
gradle assembleDebug

# APK at app/build/outputs/apk/debug/app-debug.apk
```

### Option B — Google Colab (self-contained)

Open [`scripts/build_colab.ipynb`](scripts/build_colab.ipynb) in Colab and run
all cells. It installs JDK 17, Android SDK 34, Gradle 8.7, downloads the models
(**with retry + min-byte-size check** — the 125 MB `fm_decoder` can truncate),
converts the features, and runs `assembleDebug`.

> **Critical:** `JAVA_HOME` must be set to Java 17 and **exported to the Gradle
> subprocess**. AGP 8.5 requires Java 17; Colab defaults to Java 11, which fails.

---

## Using the app

1. **Record** 🔴 (AudioRecord, mono 44.1 kHz → auto-resample) or **Import** 📁
   (MediaExtractor + MediaCodec: mp3/m4a/ogg/flac/wav → mono float).
2. Confirm the **reference transcript** (what the recording says).
3. Type your **target Indic text**.
4. Optionally tune **steps** (8–32), **guidance** (0.5–3.0), **speed** (0.5–1.5).
5. Tap **✨ Synthesize** — progress shows `step x/N` with RTF + time.
6. **▶ Play result**, then **💾 Save** to MediaStore under `Music/DhVaani`.

The app works fully offline after the first model load.

---

## Architecture / DSP notes

- **Resampler:** windowed-sinc, Kaiser window (β≈12), normalised by the sum of taps —
  matches `scipy.signal.resample_poly` to ~0.1% relative error.
- **Log-mel (Vocos):** 1024 FFT / 256 hop / 100 mels, zero-pad N_FFT/2, periodic-Hann.
- **Flow matching:** Euler, `t_shift = 0.5`, seeded noise (`seed = 666`), CFG baked into `fm_decoder`.
- **Vocoder:** backbone (ONNX) → linear head → mag+phase → overlap-add ISTFT (pad='same').
- Constants: `SR=24000`, `N_FFT=1024`, `HOP=256`, `N_MELS=100`, `FEAT_SCALE=0.1`, `TARGET_RMS=0.1`.

---

## ⚠️ Honest limitations (please read before publishing)

1. **APK ≈ 250 MB+** (int8 models ~180 MB) — beyond Play's 200 MB/file limit.
   For publishing use an **App Bundle + asset packs** or a **models-on-first-launch**
   download.
2. **Only Devanagari / Indic script renders well.** No Latin "Hinglish", digits,
   abbreviations or foreign words (chars not in the vocab are dropped). No
   language auto-detect or number normalisation.
3. **Character-level, script-dependent tokenizer** (1058 chars).
4. **Speed:** CPU only, ~5–15× realtime. Use 8 steps + short text for quick results.
   On 2 desktop CPU cores RTF ≈ 2.7; on a mid-range phone expect 5–15× slower than
   real-time. It will feel slow.
5. **Clone quality depends on the reference:** best with clean, single-speaker,
   3–8 s audio. Noisy/quiet/reverb degrades. Quiet references get RMS-boosted
   (which also amplifies background noise).
6. **24 kHz, 100-mel, Vocos.** Clean but not studio-grade. No emotion / SSML / pause control.
7. **Long text → memory grows.** Split into sentences.
8. **License respect:** Apache-2.0 source; also respect the training corpora
   (IndicTTS, Rasa, IISc SYSPIN).

---

## Ethics / consent

Voice cloning is sensitive. The app shows a prominent consent + anti-impersonation
notice and labels output as **"Synthesized by DhVaani"**. **Only clone a voice you
own or have explicit permission to use.** Cloning someone's voice without consent
can enable impersonation and is illegal in many jurisdictions.

---

## License

- Model: Apache-2.0 (`ARTPARK-IISc/DhVaani-0.5`, `k2-fsa/ZipVoice`, `charactr/vocos-mel-24khz`).
- App code in this repository: Apache-2.0 (see `LICENSE`).

> This project ships the ONNX graphs as **downloaded assets**, not committed
> binaries, to respect model sizes and licensing.
