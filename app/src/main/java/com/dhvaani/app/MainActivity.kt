package com.dhvaani.app

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.view.View
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
import com.dhvaani.app.onnx.ModelManager
import com.dhvaani.app.onnx.OnnxEngine
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

    private var engine: OnnxEngine? = null
    private var synthesizer: Synthesizer? = null
    private var modelManager: ModelManager? = null

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

        setupSpinner()
        setupSliders()
        setupListeners()

        // Kick off model loading off the UI thread immediately.
        status(getString(R.string.status_loading_models))
        executor.execute { loadModels() }
    }

    // ------------------------------------------------------------------
    // UI setup
    // ------------------------------------------------------------------
    private fun setupSpinner() {
        val items = listOf("8", "12", "16", "20", "24", "32")
        binding.spinnerSteps.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, items)
        binding.spinnerSteps.setSelection(0) // default = 8 (fast preset)
    }

    private fun setupSliders() {
        binding.sliderGuidance.valueFrom = 0.5f
        binding.sliderGuidance.valueTo = 3.0f
        binding.sliderGuidance.stepSize = 0.1f
        binding.sliderGuidance.value = 1.0f

        binding.sliderSpeed.valueFrom = 0.5f
        binding.sliderSpeed.valueTo = 1.5f
        binding.sliderSpeed.stepSize = 0.05f
        binding.sliderSpeed.value = 1.0f
    }

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

        val steps = binding.spinnerSteps.selectedItem?.toString()?.toIntOrNull() ?: 8
        val guidance = binding.sliderGuidance.value
        val speed = binding.sliderSpeed.value
        val syn = synthesizer
        if (syn == null) { toast(getString(R.string.err_models_missing)); return }

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
        binding.btnSynthesize.isEnabled = true
        binding.progress.visibility = View.GONE
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
    private fun loadModels() {
        try {
            val mgr = ModelManager(this)
            modelManager = mgr
            val models = mgr.prepare()
            if (!models.ready) {
                runOnUiThread {
                    status(models.message)
                    binding.btnDownloadModels.visibility = View.VISIBLE
                    binding.btnSynthesize.isEnabled = false
                }
                // Auto-start the download so the user usually doesn't need to tap.
                downloadModels()
                return
            }
            val eng = OnnxEngine(models.encoderPath, models.fmDecoderPath, models.vocoderBackbonePath)
            val tokenizer = mgr.loadTokenizer()
            val melFb = mgr.loadMelFb()
            val vocosHead = mgr.loadVocosHead()
            val frontend = VocosFrontend(melFb.fb, melFb.window, melFb.nFft, melFb.hop, melFb.nMels)
            val vocoder = VocosVocoder(eng, vocosHead)
            val syn = Synthesizer(eng, tokenizer, frontend, vocoder)
            runOnUiThread {
                engine = eng
                synthesizer = syn
                status(getString(R.string.status_ready))
                binding.btnDownloadModels.visibility = View.GONE
                binding.btnSynthesize.isEnabled = true
            }
        } catch (e: Exception) {
            runOnUiThread { fail(getString(R.string.err_runtime, e.message ?: "load")) }
        }
    }

    private fun downloadModels() {
        val mgr = modelManager ?: return
        if (downloading) return
        downloading = true
        // UI setup on the main thread (this can be called from the bg executor).
        runOnUiThread {
            binding.btnDownloadModels.isEnabled = false
            binding.progress.visibility = View.VISIBLE
            binding.progress.max = 100
            status(getString(R.string.status_downloading, 0, 1, "", 0))
        }

        executor.execute {
            val ok = mgr.downloadMissingModels { idx, count, name, frac ->
                val line = getString(R.string.status_downloading, idx, count, name, (frac * 100).toInt())
                runOnUiThread {
                    status(line)
                    binding.progress.progress = (frac * 100).toInt()
                }
            }
            runOnUiThread {
                binding.progress.visibility = View.GONE
                downloading = false
                if (ok) {
                    binding.btnDownloadModels.visibility = View.GONE
                    status(getString(R.string.status_download_done))
                    // Re-attempt full model load (now ready).
                    loadModels()
                } else {
                    binding.btnDownloadModels.isEnabled = true
                    status(getString(R.string.status_download_failed))
                }
            }
        }
    }

    // ------------------------------------------------------------------
    private fun status(s: String) { binding.statusLine.text = s }

    private fun toast(s: String) { Toast.makeText(this, s, Toast.LENGTH_LONG).show() }

    override fun onDestroy() {
        super.onDestroy()
        recorder.stop()
        executor.shutdown()
        try { engine?.close() } catch (_: Exception) {}
    }
}
