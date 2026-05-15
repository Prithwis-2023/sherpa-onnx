# Baymax Real-Time Speech Interaction Workflow

This document describes the complete workflow of the Baymax speech interaction algorithm implemented in `baymax.py`.

## System Architecture

```mermaid
graph TD
    A["🚀 Initialization"] -->|Initialize Models| B["ASR Recognizer<br/>SenseVoice"]
    A -->|Initialize Models| C["Speech Denoiser<br/>GTCRN"]
    A -->|Initialize Models| D["TTS Engines<br/>Pocket-TTS EN<br/>Supertonic-TTS KO"]
    
    A -->|Load Resources| E["Load Reference<br/>Audio"]
    A -->|Setup Audio| F["Start Playback<br/>Thread<br/>24kHz"]
    A -->|Setup Audio| G["Setup VAD<br/>Voice Activity<br/>Detector"]
    
    H["🎙️ Main Listen Loop"] -->|Read Samples| I["Microphone Input<br/>48kHz"]
    
    I -->|Resample| J["Audio Buffer<br/>16kHz"]
    
    J -->|Check Flag| K{"Is Baymax<br/>Speaking?"}
    K -->|Yes| L["Skip Audio<br/>Discard"]
    L --> H
    
    K -->|No| M["Add to VAD<br/>Buffer"]
    
    M -->|Accumulate| N["VAD Processing<br/>Window Size Check"]
    
    N -->|Speech Detected| O["Pop Latest<br/>Segment"]
    
    O -->|If Valid| P["Speech Denoising<br/>Clean Audio"]
    
    P -->|Clean Speech| Q["ASR Transcription<br/>Text Recognition"]
    
    Q -->|Extract Text| R{"Text Length<br/>> 1?"}
    
    R -->|Yes| S["Print: User: text"]
    S -->|Call| T["baymax_say<br/>Function"]
    
    R -->|No| U["Clear Buffer<br/>Continue Listening"]
    
    T -->|Analyze| V{"Language<br/>Detection"}
    V -->|Korean Chars| W["Set Lang = 'ko'"]
    V -->|English Chars| X["Set Lang = 'en'"]
    V -->|Other| Z["Default Lang = 'en'"]
    
    W -->|Set Flag| AA["is_speaking = True"]
    X -->|Set Flag| AA
    Z -->|Set Flag| AA
    
    AA -->|Clear Queue| AB["Initialize<br/>TTS Variables"]
    
    AB -->|Spawn Thread| AC["Run TTS Generation<br/>in Background"]
    
    AC -->|Create Config| AD{"Language<br/>Specific?"}
    
    AD -->|Korean| AE["Supertonic TTS<br/>Speaker ID=1<br/>Steps=12"]
    AD -->|English| AF["Pocket TTS<br/>Reference Audio<br/>Steps=5"]
    
    AE -->|Generate| AG["Audio Callback<br/>Process Chunks"]
    AF -->|Generate| AG
    
    AG -->|Resample if<br/>Needed| AH["Put Chunk in<br/>TTS Queue"]
    
    AH -->|Queue Data| AI["Playback Thread<br/>Pulls from Queue<br/>24kHz Output"]
    
    AI -->|Audio Out| AJ["Speaker/Audio<br/>Output"]
    
    AG -->|When Done| AK["Set tts_stopped=True"]
    
    AK -->|Signal| AL["Wait for<br/>Playback End"]
    
    AL -->|Complete| AM["is_speaking = False"]
    
    AM -->|Resume| U
    
    style A fill:#2E7D32,color:#fff
    style H fill:#1565C0,color:#fff
    style T fill:#C2185B,color:#fff
    style AG fill:#6A1B9A,color:#fff
    style AJ fill:#F57C00,color:#fff
```

## Component Overview

### Initialization Phase
- **ASR Recognizer**: SenseVoice model for multi-language speech recognition (supports Chinese, English, Japanese, Korean, Cantonese)
- **Speech Denoiser**: GTCRN model for noise reduction from microphone input
- **TTS Engines**: 
  - Pocket-TTS for English (voice cloning with reference audio)
  - Supertonic-TTS for Korean (multi-speaker support)
- **VAD (Voice Activity Detector)**: Silero VAD for detecting speech segments
- **Playback Thread**: Persistent 24kHz audio stream to avoid driver handshake delays

### Audio Processing Pipeline

1. **Input**: Microphone at 48kHz
2. **Resample**: Convert to 16kHz for processing
3. **VAD**: Detect speech activity and buffer audio
4. **Denoising**: Clean audio using GTCRN model
5. **ASR**: Transcribe to text using SenseVoice
6. **Language Detection**: Analyze text for Korean/English characters
7. **TTS Generation**: Generate response in detected language (threaded, non-blocking)
8. **Resampling**: Convert to 24kHz for consistent playback
9. **Playback**: Stream audio chunks in real-time

### Key Features

- **Echo Prevention**: `is_speaking` flag prevents the microphone from picking up Baymax's own output
- **Multi-threaded**: TTS generation runs in background thread for responsive listening
- **Dual-language Support**: Automatic language detection and model swapping
- **Memory Optimization**: Manual garbage collection when swapping TTS models
- **Real-time Streaming**: Queue-based audio chunk processing for low-latency response

### Threading Model

- **Main Thread**: Handles microphone input, VAD, denoising, and ASR
- **Playback Thread**: Dedicated stream for audio output (always running)
- **TTS Generation Thread**: Spawned per utterance for non-blocking speech generation

## Configuration

### Audio Parameters
- Microphone Input: 48 kHz
- Processing: 16 kHz
- Playback: 24 kHz
- Block Size: 1024 samples

### Model Paths
- VAD: `./silero_vad.onnx`
- ASR: `./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`
- Denoiser: `./gtcrn_simple.onnx`
- Pocket-TTS: `./sherpa-onnx-pocket-tts-int8-2026-01-26/`
- Supertonic-TTS: `./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/`

### TTS Configuration
- **English**: Reference-based voice cloning (5 generation steps)
- **Korean**: Supertonic with female speaker (12 generation steps)
