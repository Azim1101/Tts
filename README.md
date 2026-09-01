# DhVaani — Android Hindi Voice Cloner (MNN Backend, Fully Offline)

An on-device, zero-shot **voice cloning** + **text-to-speech** app for Android,
in a Hindi (English-fallback) UI. You record or import a reference voice ~3–8 s,
type what it says, then type any Indic text — the app speaks it in the reference
speaker's voice at 24 kHz. **No server, no audio ever leaves the device.**

Built on the updated **DhVaani-0.5 MNN (v1.1)** model suite and Alibaba's **MNN 3.6.1**
inference engine with native ARM NEON fp16 acceleration. **Apache-2.0.**

---

## What's inside

| Path | Purpose |
|---|---|
| `app/` | Android application (Kotlin, Material3, ViewBinding) |
| `dhvaani/` | Reusable TTS library module with native C++ engine + JNI bridge + Kotlin API |
| `dhvaani/src/main/cpp/dhvaani.cpp` | High-performance C++ DSP: windowed-sinc resampler, log-mel STFT, flow matching, ISTFT vocoder |
| `dhvaani/src/main/cpp/dhvaani_jni.cpp` | JNI bridge connecting Kotlin to C++ MNN Express Module |
| `dhvaani/src/main/java/zone/dhvaani/tts/DhVaani.kt` | Clean Kotlin TTS API (`fromAssets`, `fromDirectory`, `setPrompt`, `synthesize`, `play`) |
| `app/src/main/java/com/dhvaani/app/model/ModelDownloader.kt` | In-app streaming model downloader with retry & size verification |
| `app/src/main/java/com/dhvaani/app/audio/` | AudioRecord recorder, MediaCodec importer, AudioTrack player, WAV/MediaStore saver |
| `scripts/fetch_mnn.sh` | Fetches MNN 3.6.1 prebuilt `.so` libraries and headers |
| `scripts/download_models.sh` | Downloads DhVaani MNN graphs and `.bin` feature files |
| `.github/workflows/build-apk.yml` | GitHub Actions CI: build the debug APK and upload it as an artifact |

---

## Model assets

The MNN graphs and feature files come from
`https://huggingface.co/Bbkblo/DhVaani-0.5-MNN`:

| File | Role | Size |
|---|---|---|
| `text_encoder_int8.mnn` | text → `text_condition` | 5.8 MB |
| `fm_decoder_int8.mnn` | flow-matching velocity `(t, x, cond) → v` | 119 MB |
| `vocoder_backbone.mnn` | mels → hidden | 13 MB |
| `mel_fb.bin` | mel filterbank + window (flat float32) | 205 KB |
| `vocos_head.bin` | vocos output layer + window | 2.1 MB |
| `tokens.txt` | vocabulary | 8 KB |
| `model.json` | architecture config | 0.7 KB |

Total runtime asset footprint: **~140 MB**.

---

## Why MNN (v1.1)?

1. **Numerically faithful INT8 weights:** Quantized directly from fp32 with asymmetric INT8 weight packing, achieving **+0.99997 correlation** with fp32 PyTorch reference.
2. **Subgraphs via MNN Express:** Runs complex dynamic graphs through `MNN::Express::Module` with native C++ integration.
3. **NEON / FP16 acceleration:** `arm64-v8a` runs with `-march=armv8.2-a+fp16+dotprod` for significantly lower latency and power consumption.
4. **All-in-C++ DSP:** Mel filterbank, windowed-sinc resampler, radix-2 FFT, and overlap-add ISTFT run natively in C++ for maximum speed.

---

## Build

### Option A — Local (Android Studio or CLI)

Requirements: **JDK 17**, **Android SDK 34**, **NDK r26+ / r27**, **CMake 3.22.1**, **Gradle 8.7**, `python3`, and `curl`.

```bash
# 1. Fetch MNN 3.6.1 prebuilts & headers
bash scripts/fetch_mnn.sh

# 2. Download MNN models into assets (optional for bundling into APK)
bash scripts/download_models.sh

# 3. Build debug APK
./gradlew assembleDebug

# APK will be at: app/build/outputs/apk/debug/app-debug.apk
```

### Option B — GitHub Actions (CI)

Push to your repository branch and GitHub Actions will automatically:
1. Set up Android SDK, NDK, and CMake
2. Run `fetch_mnn.sh` and `download_models.sh`
3. Build the debug APK with `gradle assembleDebug`
4. Attach `app-debug.apk` as a downloadable **artifact**.

---

## Using the app

1. **Record** 🔴 (AudioRecord, mono 44.1 kHz → auto-resample to 24 kHz) or **Import** 📁 (MediaExtractor + MediaCodec: mp3/m4a/ogg/flac/wav → mono float).
2. Confirm the **reference transcript** (what the recording says).
3. Type your **target Indic text** (Devanagari).
4. Tap **✨ Synthesize** — progress bar shows steps in real-time.
5. **▶ Play result**, then **💾 Save** to MediaStore under `Music/DhVaani`.

On **first launch**, if model assets are not bundled in the APK, the app downloads them directly to private app storage (`filesDir/dhvaani/`) and works fully offline thereafter.

---

## Kotlin Library API (`zone.dhvaani.tts.DhVaani`)

```kotlin
// 1. Initialize engine
val tts = DhVaani.fromAssets(context, precision = DhVaani.Precision.LOW)
check(tts.isReady) { tts.lastError() }

// 2. Set reference voice
val (pcm, sr) = DhVaani.readWav(File(filesDir, "prompt.wav"))
tts.setPrompt(pcm, sr, "reference text spoken in prompt")

// 3. Configure parameters
tts.configure(numStep = 8, guidanceScale = 1.0f, speed = 1.0f, seed = 666)
tts.warmup()

// 4. Synthesize (on a background thread)
val audio: FloatArray? = tts.synthesize("नमस्ते दुनिया") { stage, cur, total ->
    Log.d("DhVaani", "$stage $cur/$total")
    true
}

// 5. Play or save
audio?.let {
    tts.play(it)
    DhVaani.writeWav(File(context.cacheDir, "output.wav"), it)
}

// 6. Release
tts.close()
```

---

## Ethics / Consent

Voice cloning is sensitive. **Only clone a voice you own or have explicit permission to use.** Cloning someone's voice without consent can enable impersonation and is illegal in many jurisdictions.

---

## License

- Models: Apache-2.0 (`ARTPARK-IISc/DhVaani-0.5`, `Bbkblo/DhVaani-0.5-MNN`, `k2-fsa/ZipVoice`, `charactr/vocos-mel-24khz`).
- MNN: Apache-2.0 (`alibaba/MNN`).
- App code: Apache-2.0 (see `LICENSE`).
