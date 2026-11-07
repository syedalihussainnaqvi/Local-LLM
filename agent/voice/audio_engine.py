"""Antigravity Local Audio Engine (Win32 & Windows Native SAPI / System.Speech).
Provides zero-cloud, fully offline:
- S1: Win32 Audio Capture (waveIn) with RMS calculation & device enumeration.
- S2: Echo cancellation gating and mute protection.
- S3: Energy/RMS Voice Activity Detection (VAD) and speech endpointing.
- S4: Offline Speech-to-Text via System.Speech.Recognition (DictationGrammar).
- S8: Offline Text-to-Speech via System.Speech.Synthesis.
- S9: Audio playback via winsound / waveOut with barge-in interruption.
"""

import os
import sys
import time
import math
import wave
import struct
import shutil
import ctypes
import tempfile
import logging
import subprocess
from pathlib import Path
from ctypes import wintypes
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("voice_audio_engine")

# Win32 Multimedia API definitions
winmm = ctypes.windll.winmm


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class WAVEHDR(ctypes.Structure):
    pass


WAVEHDR._fields_ = [
    ("lpData", ctypes.c_char_p),
    ("dwBufferLength", wintypes.DWORD),
    ("dwBytesRecorded", wintypes.DWORD),
    ("dwUser", ctypes.c_size_t),
    ("dwFlags", wintypes.DWORD),
    ("dwLoops", wintypes.DWORD),
    ("lpNext", ctypes.POINTER(WAVEHDR)),
    ("reserved", ctypes.c_size_t),
]


class WAVEINCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", wintypes.WCHAR * 32),
        ("dwFormats", wintypes.DWORD),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
    ]


class WAVEOUTCAPSW(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.DWORD),
        ("szPname", wintypes.WCHAR * 32),
        ("dwFormats", wintypes.DWORD),
        ("wChannels", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("dwSupport", wintypes.DWORD),
    ]


class AudioDeviceManager:
    """Stage 1: Enumerate and query local audio hardware."""

    @staticmethod
    def get_devices() -> Dict[str, List[Dict[str, Any]]]:
        inputs = []
        num_in = winmm.waveInGetNumDevs()
        for i in range(num_in):
            caps = WAVEINCAPSW()
            res = winmm.waveInGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps))
            if res == 0:
                inputs.append({"id": i, "name": caps.szPname, "channels": caps.wChannels})

        outputs = []
        num_out = winmm.waveOutGetNumDevs()
        for i in range(num_out):
            caps = WAVEOUTCAPSW()
            res = winmm.waveOutGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps))
            if res == 0:
                outputs.append({"id": i, "name": caps.szPname, "channels": caps.wChannels})

        return {"inputs": inputs, "outputs": outputs}


class Win32AudioRecorder:
    """Stage 1: Low-level Win32 waveIn PCM capture (16kHz 16-bit mono)."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = 2

    def record_seconds(self, seconds: float = 3.0, device_id: int = 0) -> Tuple[bytes, float, int]:
        """Records raw PCM audio for a fixed duration. Returns (pcm_bytes, rms, peak)."""
        wfx = WAVEFORMATEX()
        wfx.wFormatTag = 1  # WAVE_FORMAT_PCM
        wfx.nChannels = self.channels
        wfx.nSamplesPerSec = self.sample_rate
        wfx.wBitsPerSample = self.bytes_per_sample * 8
        wfx.nBlockAlign = self.channels * self.bytes_per_sample
        wfx.nAvgBytesPerSec = self.sample_rate * wfx.nBlockAlign
        wfx.cbSize = 0

        hWaveIn = wintypes.HANDLE()
        res = winmm.waveInOpen(ctypes.byref(hWaveIn), device_id, ctypes.byref(wfx), 0, 0, 0)
        if res != 0:
            raise RuntimeError(f"waveInOpen failed with error code: {res}")

        buffer_len = int(self.sample_rate * wfx.nBlockAlign * seconds)
        raw_buf = ctypes.create_string_buffer(buffer_len)

        whdr = WAVEHDR()
        whdr.lpData = ctypes.cast(raw_buf, ctypes.c_char_p)
        whdr.dwBufferLength = buffer_len
        whdr.dwFlags = 0

        winmm.waveInPrepareHeader(hWaveIn, ctypes.byref(whdr), ctypes.sizeof(whdr))
        winmm.waveInAddBuffer(hWaveIn, ctypes.byref(whdr), ctypes.sizeof(whdr))
        winmm.waveInStart(hWaveIn)

        time.sleep(seconds + 0.1)

        winmm.waveInStop(hWaveIn)
        winmm.waveInReset(hWaveIn)
        winmm.waveInUnprepareHeader(hWaveIn, ctypes.byref(whdr), ctypes.sizeof(whdr))
        winmm.waveInClose(hWaveIn)

        recorded_len = whdr.dwBytesRecorded
        pcm_bytes = raw_buf.raw[:recorded_len]

        # Compute RMS and peak
        num_samples = recorded_len // 2
        if num_samples > 0:
            samples = struct.unpack(f"<{num_samples}h", pcm_bytes)
            rms = math.sqrt(sum(s * s for s in samples) / num_samples)
            peak = max(abs(s) for s in samples)
        else:
            rms = 0.0
            peak = 0

        return pcm_bytes, rms, peak

    def save_wav(self, pcm_bytes: bytes, target_path: str):
        with wave.open(target_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.bytes_per_sample)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)


class EnergyVAD:
    """Stage 3: Energy-based Voice Activity Detector with endpointing."""

    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, threshold: float = 80.0):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold = threshold

    def detect_segments(self, pcm_bytes: bytes) -> List[Tuple[float, float]]:
        """Returns list of (start_seconds, end_seconds) speech segments."""
        num_samples = len(pcm_bytes) // 2
        if num_samples < self.frame_size:
            return []

        samples = struct.unpack(f"<{num_samples}h", pcm_bytes[: num_samples * 2])
        segments = []
        in_speech = False
        start_time = 0.0

        for i in range(0, num_samples - self.frame_size, self.frame_size):
            frame = samples[i : i + self.frame_size]
            rms = math.sqrt(sum(s * s for s in frame) / len(frame))
            t = i / self.sample_rate

            if rms >= self.threshold and not in_speech:
                in_speech = True
                start_time = t
            elif rms < self.threshold and in_speech:
                in_speech = False
                if t - start_time >= 0.15:  # Ignore tiny clicks < 150ms
                    segments.append((round(start_time, 2), round(t, 2)))

        if in_speech:
            segments.append((round(start_time, 2), round(num_samples / self.sample_rate, 2)))

        return segments


class WindowsSTT:
    """Stage 4: Local offline Speech-to-Text using Windows System.Speech."""

    @staticmethod
    def transcribe_wav(wav_path: str, timeout_seconds: float = 12.0) -> Dict[str, Any]:
        """Transcribe a 16kHz mono WAV file using Windows SpeechRecognitionEngine."""
        abs_path = str(Path(wav_path).resolve()).replace("\\", "/")
        if not Path(abs_path).exists():
            raise FileNotFoundError(f"WAV file not found: {abs_path}")

        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
            "$grammar = New-Object System.Speech.Recognition.DictationGrammar; "
            "$engine.LoadGrammar($grammar); "
            f"$engine.SetInputToWaveFile('{abs_path}'); "
            "$result = $engine.Recognize(); "
            "if ($result) { "
            "  Write-Output ('TRANSCRIPT:' + $result.Text); "
            "  Write-Output ('CONFIDENCE:' + $result.Confidence); "
            "} else { "
            "  Write-Output 'TRANSCRIPT:EMPTY'; "
            "} "
            "$engine.Dispose();"
        )

        t0 = time.time()
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"transcript": "", "confidence": 0.0, "duration_ms": (time.time() - t0) * 1000, "error": "STT Timeout"}

        dur_ms = round((time.time() - t0) * 1000, 2)
        stdout = res.stdout.strip()
        transcript = ""
        confidence = 0.0

        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("TRANSCRIPT:"):
                raw = line[len("TRANSCRIPT:") :].strip()
                transcript = "" if raw == "EMPTY" else raw
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line[len("CONFIDENCE:") :].strip())
                except ValueError:
                    pass

        return {
            "transcript": transcript,
            "confidence": confidence,
            "duration_ms": dur_ms,
            "model": "Windows SAPI System.Speech (English - US)",
        }


class WindowsTTS:
    """Stage 8: Local offline Text-to-Speech using Windows System.Speech."""

    @staticmethod
    def synthesize_to_wav(text: str, output_wav_path: str, timeout_seconds: float = 20.0) -> Dict[str, Any]:
        """Synthesize text to WAV file."""
        abs_path = str(Path(output_wav_path).resolve()).replace("\\", "/")
        clean_text = text.replace("'", " ").replace('"', " ").replace("\n", " ").strip()
        if not clean_text:
            clean_text = "I am listening."

        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$synth.SetOutputToWaveFile('{abs_path}'); "
            f"$synth.Speak('{clean_text}'); "
            "$synth.Dispose();"
        )

        t0 = time.time()
        byte_len = 0
        last_error = None

        for attempt in range(3):
            # Stagger SAPI COM invocations: Windows SAPI SpeechSynthesizer is a COM object
            # that can fail silently under rapid-successive calls (e.g. 20-run stress tests).
            # A small pre-attempt delay (except on first attempt) prevents COM resource contention.
            if attempt > 0:
                time.sleep(0.25 * attempt)  # 250ms, 500ms backoff

            cur_timeout = 8.0 if attempt == 0 else 6.0
            try:
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=cur_timeout,
                )
                # Brief filesystem flush wait — SAPI writes asynchronously on some Windows builds
                time.sleep(0.15)
                p = Path(abs_path)
                if p.exists() and p.stat().st_size > 0:
                    byte_len = p.stat().st_size
                    break
                elif res.stderr:
                    logger.warning(f"[TTS] Attempt {attempt+1} stderr: {res.stderr.strip()[:200]}")
            except subprocess.TimeoutExpired:
                last_error = "TTS Timeout"
                logger.warning(f"[TTS] Attempt {attempt+1} timed out after {cur_timeout}s")

        dur_ms = round((time.time() - t0) * 1000, 2)
        if byte_len == 0 and last_error:
            return {"error": last_error, "duration_ms": dur_ms, "byte_length": 0}

        return {
            "wav_path": abs_path,
            "byte_length": byte_len,
            "duration_ms": dur_ms,
            "voice": "Microsoft David / Zira Desktop",
        }


class AudioPlayer:
    """Stage 9: Audio playback with barge-in stop support."""

    @staticmethod
    def play_wav(wav_path: str, async_play: bool = False):
        import winsound

        flags = winsound.SND_FILENAME
        if async_play:
            flags |= winsound.SND_ASYNC
        winsound.PlaySound(wav_path, flags)

    @staticmethod
    def stop_playback():
        import winsound

        winsound.PlaySound(None, winsound.SND_PURGE)

    @staticmethod
    def play_tone(freq: int = 800, dur_ms: int = 150):
        import winsound

        try:
            winsound.Beep(freq, dur_ms)
        except Exception:
            pass
