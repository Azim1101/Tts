package com.dhvaani.app

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.dhvaani.app.audio.AudioImporter
import com.dhvaani.app.audio.AudioPlayer
import com.dhvaani.app.audio.AudioRecorder
import com.dhvaani.app.audio.ImportedAudio
import com.dhvaani.app.audio.MediaStoreSaver
import com.dhvaani.app.databinding.ActivityMainBinding
import com.dhvaani.app.dsp.VocosFrontend
import com.dhvaani.app.dsp.VocosVocoder
import com.dhvaani.app.model.ModelCatalog
import com.dhvaani.app.model.ModelSpec
import com.dhvaani.app.onnx.ModelEngine
import com.dhvaani.app.onnx.ModelManager
import com.dhvaani.app.onnx.MnnEngine
import com.dhvaani.app.tts.Synthesizer
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val executor = Executors.newSingleThreadExecutor()

    private val recorder = AudioRecorder()
    private val player = AudioPlayer()

    private var reference: FloatArray? = null
    private var refSampleRate: Int = 44100
    private var result: FloatArray? = null
    private var busy = false
    @Volatile private var downloading = false

    // Current model selection. MNN-only build.
    private val catalog: List<ModelSpec> get() = ModelCatalog.ALL
    private var selectedSpec: ModelSpec = ModelCatalog.DHVAANI_MNN
    private var currentSpec: ModelSpec? = null

    private var engine: ModelEngine? = null
    private var synthesizer: Synthesizer? = null
    private var modelManager: ModelManager? = null

    private var spinnerListener: AdapterView.OnItemSelectedListener? = null

    // Hold the current Toast so a new one cancels the previous. This prevents the
    // stale "Model files missing..." toast from lingering on screen after the
    // auto-download completes and the status becomes "Ready" again.
    private var currentToast: Toast? = null

    // ------------------------------------------------------------------
    // Synthesis parameters — hard-coded in v0.7.5 (the steps/guidance/speed
    // sliders were removed for the simpler one-tap-clone UI). These match
    // the values the README and the original `dhvaani_torchfree.py`
    // reference script treat as the "fast preset".
    // ------------------------------------------------------------------
    private val steps: Int = 8
    private val guidance: Float = 1.0f
    private val speed: Float = 1.0f

    private val recordPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startRecording() else toast(getString(R.string.err_empty_ref))
        }

    private val importLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) importAudio(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnSynthesize.isEnabled = false

        setupListeners()
        setupModelSelector()

        // Kick off model loading of the default model off the UI thread.
        status(getString(R.string.status_loading_models))
        executor.execute { loadModels(selectedSpec) }
    }

    // ------------------------------------------------------------------
    // UI setup
    // ------------------------------------------------------------------

    private fun setupListeners() {
        binding.btnRecord.setOnClickListener { toggleRecord() }

        binding.btnPlayRef.setOnClickListener {
            val ref = reference
            if (ref == null) { toast(getString(R.string.err_empty_ref)); return@setOnClickListener }
            player.play(ref, refSampleRate)
        }

        binding.btnImport.setOnClickListener {
            importLauncher.launch(arrayOf("audio/*"))
        }

        binding.btnSynthesize.setOnClickListener { synthesize() }

        binding.btnDownloadModels.setOnClickListener { downloadModels() }

        binding.btnPlayResult.setOnClickListener {
            val r = result
            if (r == null) { toast(getString(R.string.err_empty_target)); return@setOnClickListener }
            player.play(r, com.dhvaani.app.dsp.DspConstants.SR)
        }

        binding.btnSave.setOnClickListener { saveResult() }
    }

    private fun setupModelSelector() {
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            catalog.map { it.title }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.modelSpinner.adapter = adapter

        spinnerListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position >= 0 && position < catalog.size) {
                    onModelSelected(catalog[position])
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        binding.modelSpinner.onItemSelectedListener = spinnerListener
    }

    private fun onModelSelected(spec: ModelSpec) {
        if (spec.id == selectedSpec.id) return

        selectedSpec = spec
        if (busy || downloading) {
            status(getString(R.string.status_wait_model_switch, spec.title))
            return
        }
        status(getString(R.string.status_model_selected, spec.title))
        executor.execute { loadModels(spec) }
    }

    // ------------------------------------------------------------------
    // Recording
    // ------------------------------------------------------------------
    private fun toggleRecord() {
        if (busy) { toast(getString(R.string.status_synthesizing)); return }
        if (recorder.isRecording) {
            val samples = recorder.stop()
            if (samples.size > 0) {
                reference = samples
                refSampleRate = 44100
                binding.btnRecord.text = getString(R.string.btn_record)
                binding.btnPlayRef.isEnabled = true
                status(getString(R.string.status_ready) + " (${samples.size / 44100}s)")
            }
            return
        }
        checkRecordPermissionAndStart()
    }

    private fun checkRecordPermissionAndStart() {
        val perm = Manifest.permission.RECORD_AUDIO
        if (ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED) {
            startRecording()
        } else {
            recordPermission.launch(perm)
        }
    }

    private fun startRecording() {
        recorder.start()
        binding.btnRecord.text = getString(R.string.btn_stop)
        status(getString(R.string.status_recording))
    }

    // ------------------------------------------------------------------
    // Import
    // ------------------------------------------------------------------
    private fun importAudio(uri: Uri) {
        status(getString(R.string.status_recording))
        executor.execute {
            try {
                val imported: ImportedAudio = AudioImporter(this).decode(uri)
                runOnUiThread {
                    reference = imported.samples
                    refSampleRate = imported.sampleRate
                    binding.btnPlayRef.isEnabled = true
                    status(getString(R.string.status_ready) + " (${imported.samples.size / imported.sampleRate}s, ${imported.sampleRate}Hz)")
                }
            } catch (e: Exception) {
                runOnUiThread { toast(getString(R.string.err_runtime, e.message ?: "decode")) }
            }
        }
    }

    // ------------------------------------------------------------------
    // Synthesis
    // ------------------------------------------------------------------
    private fun synthesize() {
        if (busy) return
        val ref = reference ?: run { toast(getString(R.string.err_empty_ref)); return }
        val transcript = binding.refTranscript.text?.toString() ?: ""
        if (transcript.isBlank()) { toast(getString(R.string.err_empty_transcript)); return }
        val target = binding.targetText.text?.toString() ?: ""
        if (target.isBlank()) { toast(getString(R.string.err_empty_target)); return }

        val syn = synthesizer
        if (syn == null) {
            binding.btnSynthesize.isEnabled = false
            binding.btnDownloadModels.visibility = View.VISIBLE
            binding.btnDownloadModels.isEnabled = true
            status(getString(R.string.err_models_missing))
            toast(getString(R.string.err_models_missing))
            return
        }

        busy = true
        binding.btnSynthesize.isEnabled = false
        binding.progress.visibility = View.VISIBLE
        binding.progress.max = steps
        val start = SystemClock.elapsedRealtime()
        status(getString(R.string.status_synthesizing, 0, steps))

        executor.execute {
            try {
                val audio = syn.synthesize(
                    reference = ref,
                    refSampleRate = refSampleRate,
                    refTranscript = transcript,
                    targetText = target,
                    steps = steps,
                    guidance = guidance,
                    speed = speed
                ) { done, total ->
                    runOnUiThread {
                        binding.progress.progress = done
                        status(getString(R.string.status_synthesizing, done, total))
                    }
                }
                val elapsed = SystemClock.elapsedRealtime() - start
                val seconds = elapsed / 1000f
                val audioSecs = audio.size.toFloat() / com.dhvaani.app.dsp.DspConstants.SR
                val rtf = if (audioSecs > 0f) seconds / audioSecs else 0f
                runOnUiThread {
                    result = audio
                    busy = false
                    binding.btnSynthesize.isEnabled = true
                    binding.btnPlayResult.isEnabled = true
                    binding.btnSave.isEnabled = true
                    binding.progress.visibility = View.GONE
                    status(getString(R.string.status_done, "%.2fs".format(seconds), "%.2f".format(rtf)))
                }
            } catch (e: IllegalArgumentException) {
                val msg = when (e.message) {
                    "NO_TOKENS" -> getString(R.string.err_no_tokens)
                    "EMPTY_TRANSCRIPT" -> getString(R.string.err_empty_transcript)
                    "REF_LONGER" -> getString(R.string.err_ref_longer)
                    else -> getString(R.string.err_runtime, e.message ?: "unknown")
                }
                runOnUiThread { fail(msg) }
            } catch (e: Exception) {
                runOnUiThread { fail(getString(R.string.err_runtime, e.message ?: "unknown")) }
            }
        }
    }

    private fun fail(msg: String) {
        busy = false
        binding.progress.visibility = View.GONE
        binding.btnSynthesize.isEnabled = synthesizer != null
        status(msg)
        toast(msg)
    }

    // ------------------------------------------------------------------
    // Save
    // ------------------------------------------------------------------
    private fun saveResult() {
        val r = result ?: return
        executor.execute {
            val uri = MediaStoreSaver.save(this, r, com.dhvaani.app.dsp.DspConstants.SR)
            runOnUiThread { status(getString(R.string.status_saved)); toast(getString(R.string.status_saved)) }
        }
    }

    // ------------------------------------------------------------------
    // Model loading / download
    // ------------------------------------------------------------------
    private fun loadModels(spec: ModelSpec) {
        val mgr = ModelManager(this)
        modelManager = mgr
        val models = mgr.prepare(spec)
        if (!models.ready) {
            runOnUiThread {
                status(models.message)
                binding.btnDownloadModels.visibility = View.VISIBLE
                binding.btnDownloadModels.isEnabled = true
                binding.btnSynthesize.isEnabled = false
            }
            // Auto-start the download for the selected model.
            downloadModels(spec)
            return
        }

        var built: ModelEngine? = null
        try {
            if (!MnnEngine.isRuntimeAvailable()) {
                runOnUiThread {
                    status(getString(R.string.status_mnn_runtime_missing))
                    binding.btnDownloadModels.visibility = View.VISIBLE
                    binding.btnDownloadModels.isEnabled = true
                    binding.btnSynthesize.isEnabled = false
                }
                return
            }
            built = MnnEngine(models.encoderPath, models.fmDecoderPath, models.vocoderBackbonePath)
        } catch (e: Throwable) {
            Log.e("DhVaani", "loadModels(${spec.id}) failed", e)
            runOnUiThread {
                binding.btnSynthesize.isEnabled = false
                binding.btnDownloadModels.visibility = View.VISIBLE
                binding.btnDownloadModels.isEnabled = true
                fail(getString(R.string.err_runtime, e.message ?: e::class.java.simpleName))
            }
            return
        }

        val readyEngine = built ?: return
        val syn: Synthesizer
        try {
            val tokenizer = mgr.loadTokenizer()
            val melFb = mgr.loadMelFb()
            val vocosHead = mgr.loadVocosHead()
            val frontend = VocosFrontend(melFb.fb, melFb.window, melFb.nFft, melFb.hop, melFb.nMels)
            val vocoder = VocosVocoder(readyEngine, vocosHead)
            syn = Synthesizer(readyEngine, tokenizer, frontend, vocoder)
        } catch (e: Throwable) {
            Log.e("DhVaani", "loadModels(${spec.id}) feature load failed", e)
            runCatching { readyEngine.close() }
            runOnUiThread {
                binding.btnSynthesize.isEnabled = false
                binding.btnDownloadModels.visibility = View.VISIBLE
                binding.btnDownloadModels.isEnabled = true
                fail(getString(R.string.err_runtime, e.message ?: e::class.java.simpleName))
            }
            return
        }
        runOnUiThread {
            engine = readyEngine
            synthesizer = syn
            currentSpec = spec
            status(getString(R.string.status_ready_model, spec.title))
            binding.btnDownloadModels.visibility = View.GONE
            binding.btnDownloadModels.isEnabled = true
            binding.btnSynthesize.isEnabled = true
        }
    }

    private fun downloadModels() {
        downloadModels(selectedSpec)
    }

    private fun downloadModels(spec: ModelSpec) {
        val mgr = modelManager ?: return
        if (downloading) return
        downloading = true
        runOnUiThread {
            binding.btnDownloadModels.isEnabled = false
            binding.progress.visibility = View.VISIBLE
            binding.progress.max = 100
            status(getString(R.string.status_downloading, 0, 1, spec.title, 0))
        }

        executor.execute {
            val modelsDir = java.io.File(filesDir, "models")
            if (!modelsDir.exists()) modelsDir.mkdirs()
            val loaded = try {
                com.dhvaani.app.onnx.ModelDownloader.downloadMissing(modelsDir, spec) { idx, count, name, frac ->
                    val line = getString(R.string.status_downloading, idx, count, "$name (${spec.title})", (frac * 100).toInt())
                    runOnUiThread {
                        status(line)
                        binding.progress.progress = (frac * 100).toInt()
                    }
                }
            } catch (e: Throwable) {
                Log.e("DhVaani.Download", "downloadMissing threw", e)
                false
            }
            runOnUiThread {
                binding.progress.visibility = View.GONE
                downloading = false
                if (loaded) {
                    binding.btnDownloadModels.visibility = View.GONE
                    status(getString(R.string.status_download_done))
                    // Re-attempt full model load (now ready) off the UI thread.
                    executor.execute { loadModels(spec) }
                } else {
                    binding.btnDownloadModels.isEnabled = true
                    status(getString(R.string.status_download_failed))
                }
            }
        }
    }

    // ------------------------------------------------------------------
    private fun status(s: String) {
        binding.statusLine.text = s
        currentToast?.cancel()
        currentToast = null
    }

    private fun toast(s: String) {
        currentToast?.cancel()
        val t = Toast.makeText(this, s, Toast.LENGTH_LONG)
        currentToast = t
        t.show()
    }

    override fun onDestroy() {
        super.onDestroy()
        recorder.stop()
        executor.shutdown()
        try { engine?.close() } catch (_: Exception) {}
        currentToast?.cancel()
        currentToast = null
    }
}
