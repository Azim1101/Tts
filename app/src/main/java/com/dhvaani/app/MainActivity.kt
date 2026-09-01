package com.dhvaani.app

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
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
import com.dhvaani.app.model.ModelDownloader
import java.io.File
import java.util.concurrent.Executors
import zone.dhvaani.tts.DhVaani

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

    private var tts: DhVaani? = null
    private var currentToast: Toast? = null

    // Balanced synthesis preset (8 Euler steps)
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

        // Initialize engine off the UI thread
        status(getString(R.string.status_loading_models))
        executor.execute { loadModels() }
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
            player.play(r, DhVaani.SAMPLE_RATE)
        }

        binding.btnSave.setOnClickListener { saveResult() }
    }

    // ------------------------------------------------------------------
    // Recording
    // ------------------------------------------------------------------
    private fun toggleRecord() {
        if (busy) { toast(getString(R.string.status_synthesizing, 0, steps)); return }
        if (recorder.isRecording) {
            val samples = recorder.stop()
            if (samples.isNotEmpty()) {
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

        val engine = tts
        if (engine == null || !engine.isReady) {
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
                if (!engine.setPrompt(ref, refSampleRate, transcript)) {
                    val err = engine.lastError()
                    val msg = when {
                        err.contains("no tokens", ignoreCase = true) -> getString(R.string.err_no_tokens)
                        err.contains("empty", ignoreCase = true) -> getString(R.string.err_empty_transcript)
                        else -> getString(R.string.err_runtime, err)
                    }
                    runOnUiThread { fail(msg) }
                    return@execute
                }

                engine.configure(numStep = steps, guidanceScale = guidance, speed = speed, seed = 666)

                val audio = engine.synthesize(target) { stage, cur, total ->
                    runOnUiThread {
                        if (stage == "fm_decoder") {
                            binding.progress.progress = cur
                            status(getString(R.string.status_synthesizing, cur, total))
                        }
                    }
                    true
                }

                if (audio == null) {
                    val err = engine.lastError()
                    val msg = when {
                        err.contains("no tokens", ignoreCase = true) -> getString(R.string.err_no_tokens)
                        err.contains("prompt longer", ignoreCase = true) -> getString(R.string.err_ref_longer)
                        else -> getString(R.string.err_runtime, err)
                    }
                    runOnUiThread { fail(msg) }
                    return@execute
                }

                val elapsed = SystemClock.elapsedRealtime() - start
                val seconds = elapsed / 1000f
                val audioSecs = audio.size.toFloat() / DhVaani.SAMPLE_RATE
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
            } catch (e: Exception) {
                runOnUiThread { fail(getString(R.string.err_runtime, e.message ?: "unknown")) }
            }
        }
    }

    private fun fail(msg: String) {
        busy = false
        binding.progress.visibility = View.GONE
        binding.btnSynthesize.isEnabled = (tts?.isReady == true)
        status(msg)
        toast(msg)
    }

    // ------------------------------------------------------------------
    // Save
    // ------------------------------------------------------------------
    private fun saveResult() {
        val r = result ?: return
        executor.execute {
            val uri = MediaStoreSaver.save(this, r, DhVaani.SAMPLE_RATE)
            runOnUiThread {
                if (uri != null) {
                    status(getString(R.string.status_saved))
                    toast(getString(R.string.status_saved))
                } else {
                    toast(getString(R.string.err_runtime, "Save failed"))
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Model loading / download
    // ------------------------------------------------------------------
    private fun loadModels() {
        try {
            val modelDir = File(filesDir, "dhvaani")
            val allPresentInDir = DhVaani.REQUIRED_FILES.all { File(modelDir, it).length() > 0 }

            var engine: DhVaani? = null
            if (allPresentInDir) {
                engine = DhVaani.fromDirectory(modelDir)
            } else {
                val hasAssets = runCatching {
                    assets.open("dhvaani/tokens.txt").close()
                    true
                }.getOrDefault(false)

                if (hasAssets) {
                    engine = DhVaani.fromAssets(this, "dhvaani")
                }
            }

            if (engine != null && engine.isReady) {
                tts = engine
                runOnUiThread {
                    status(getString(R.string.status_ready))
                    binding.btnDownloadModels.visibility = View.GONE
                    binding.btnSynthesize.isEnabled = true
                }
                return
            }

            runOnUiThread {
                status(getString(R.string.err_models_missing))
                binding.btnDownloadModels.visibility = View.VISIBLE
                binding.btnSynthesize.isEnabled = false
            }
            downloadModels()
        } catch (e: Throwable) {
            Log.e("DhVaani", "loadModels failed", e)
            runOnUiThread {
                binding.btnSynthesize.isEnabled = false
                binding.btnDownloadModels.visibility = View.VISIBLE
                binding.btnDownloadModels.isEnabled = true
                fail(getString(R.string.err_runtime, e.message ?: e::class.java.simpleName))
            }
        }
    }

    private fun downloadModels() {
        if (downloading) return
        downloading = true
        val modelDir = File(filesDir, "dhvaani")

        runOnUiThread {
            binding.btnDownloadModels.isEnabled = false
            binding.progress.visibility = View.VISIBLE
            binding.progress.max = 100
            status(getString(R.string.status_downloading, 0, 1, "", 0))
        }

        executor.execute {
            val ok = try {
                ModelDownloader.downloadMissing(modelDir) { idx, count, name, frac ->
                    val line = getString(R.string.status_downloading, idx, count, name, (frac * 100).toInt())
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
                if (ok) {
                    binding.btnDownloadModels.visibility = View.GONE
                    status(getString(R.string.status_download_done))
                    loadModels()
                } else {
                    binding.btnDownloadModels.isEnabled = true
                    status(getString(R.string.status_download_failed))
                }
            }
        }
    }

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
        player.stop()
        executor.shutdown()
        tts?.close()
        tts = null
        currentToast?.cancel()
        currentToast = null
    }
}
