import threading
import queue
import time
import numpy as np
import sherpa_onnx
import sounddevice as sd
from pathlib import Path
import logging
import librosa
import soundfile as sf
import re
import gc  # for manual memory cleaning

# global state for tts playback
tts_queue = queue.Queue()
tts_event = threading.Event()
tts_started = False  # set to true once generated_audio_callback is called
tts_stopped = False  # set to true once all text is processed
tts_killed = False   # set to true once exited

# global state for the current active tts
current_tts = None
active_lang = None  # 'en' or 'ko'

# configuration
# wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
VAD_MODEL = "./silero_vad.onnx"
ASR_MODEL = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"
ASR_TOKENS = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"

#wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
DENOISER_MODEL = "./gtcrn_simple.onnx"

#curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
#tar xf kokoro-multi-lang-v1_0.tar.bz2
#rm kokoro-multi-lang-v1_0.tar.bz2
KOKORO_TTS_MODEL = "./kokoro-multi-lang-v1_0/model.onnx"
KOKORO_TTS_VOICES = "./kokoro-multi-lang-v1_0/voices.bin"
KOKORO_TTS_TOKENS = "./kokoro-multi-lang-v1_0/tokens.txt"
KOKORO_TTS_DATA_DIR = "./kokoro-multi-lang-v1_0/espeak-ng-data"
KOKORO_TTS_LEXICON = "./kokoro-multi-lang-v1_0/lexicon-us-en.txt"

#wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
#tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
#rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
POCKET_LM_FLOW = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx"
POCKET_LM_MAIN = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx"
POCKET_ENCODER = "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx"
POCKET_DECODER = "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx"
POCKET_TEXT_CONDITIONER = "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx"
POCKET_VOCAB_JSON = "./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json"
POCKET_TOKENS_SCORES_JSON = "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json"  

#wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
#tar xvf sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
#rm sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2
SUPERTONIC_DURATION_PREDICTOR = "./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx"
SUPERTONIC_TEXT_ENCODER = "./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx"
SUPERTONIC_VECTOR_ESTIMATOR = "./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx"
SUPERTONIC_VOCODER = "./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx"
SUPERTONIC_TTS_JSON ="./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json"
SUPERTONIC_UNICODE_INDEXER = "./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin"
SUPERTONIC_VOICE_STYLE = "./sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin"

#wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
#tar xvf sherpa-onnx-whisper-tiny.tar.bz2
#rm sherpa-onnx-whisper-tiny.tar.bz2
WHISPER_ENCODER = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx"
WHISPER_DECODER = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx"

tts_en = None
tts_ko = None
is_speaking = False  # global flag to prevent Baymax from hearing its own echo

def resample_audio(samples, original_sr, target_sr = 24000):
    """Resample audio to the target sample rate if needed."""
    # Ensure samples is a 1D array (Mono)
    if samples.ndim > 1:
        # take only the first channel if its stereo
        samples = samples[:, 0]
    
    samples = samples.flatten().astype(np.float32)

    if original_sr == target_sr:
        return samples
    # linear interpolation for resampling
    duration = len(samples) / original_sr
    target_n_samples = int(duration * target_sr)
    return np.interp(
        np.linspace(0, duration, target_n_samples),
        np.linspace(0, duration, len(samples)),
        samples
    ).astype(np.float32)


# Initialization Functions

def create_recognizer():
    """Setup for ASR. We are using SenseVoice"""
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model = ASR_MODEL,
        tokens = ASR_TOKENS,
        num_threads = 3,
        use_itn = True,
        language = "", # one out of "zh", "en", "ja", "ko", "yue"
        debug=False,
        hr_rule_fsts = "",
        hr_lexicon = "",
    )
    

def create_speech_denoiser():
    """Setup for Speech Enhancement. We are using GTCRN model"""
    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model = sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                model = DENOISER_MODEL
            ),
        debug = False,
        num_threads = 1,
        provider = "cpu",
        )
    )
    return sherpa_onnx.OfflineSpeechDenoiser(config)


def create_kokoro_tts():
    """Setup for TTS. We are using Kokoro Multi-Lang model"""
    config = sherpa_onnx.OfflineTtsConfig(
        model = sherpa_onnx.OfflineTtsModelConfig(
            kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model = KOKORO_TTS_MODEL,
                voices = KOKORO_TTS_VOICES,
                tokens = KOKORO_TTS_TOKENS,
                data_dir = KOKORO_TTS_DATA_DIR,
                lexicon = KOKORO_TTS_LEXICON,
            ),
            provider = "cpu",
            debug = False,
            num_threads = 2,
        ),
        rule_fsts = "",
        max_num_sentences = 1,
    )
    return sherpa_onnx.OfflineTts(config)


def create_pocket_tts():
    config = sherpa_onnx.OfflineTtsConfig(
        model = sherpa_onnx.OfflineTtsModelConfig(
            pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
                lm_flow = POCKET_LM_FLOW,
                lm_main = POCKET_LM_MAIN,
                encoder = POCKET_ENCODER,
                decoder = POCKET_DECODER,
                text_conditioner = POCKET_TEXT_CONDITIONER,
                vocab_json = POCKET_VOCAB_JSON,
                token_scores_json = POCKET_TOKENS_SCORES_JSON,  
            ),
            debug = False,
            num_threads = 3,
            provider = "cpu",
        )
    )
    return sherpa_onnx.OfflineTts(config)


def create_supertonic_tts():
    config = sherpa_onnx.OfflineTtsConfig(
        model = sherpa_onnx.OfflineTtsModelConfig(
            supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
                duration_predictor = SUPERTONIC_DURATION_PREDICTOR,
                text_encoder = SUPERTONIC_TEXT_ENCODER,
                vector_estimator = SUPERTONIC_VECTOR_ESTIMATOR,
                vocoder = SUPERTONIC_VOCODER,
                tts_json = SUPERTONIC_TTS_JSON,
                unicode_indexer = SUPERTONIC_UNICODE_INDEXER,
                voice_style = SUPERTONIC_VOICE_STYLE,
            ),
            debug = False,
            num_threads = 3,
            provider = "cpu",
        )
    )
    return sherpa_onnx.OfflineTts(config)


def whisper_multilingual():
    """This function can be used to initialize the whisper multilingual model for the spoken language identification"""
    config = sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper = sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder = WHISPER_ENCODER,
            decoder = WHISPER_DECODER,
        ),
        num_threads = 3,
        debug = False,
        provider = "cpu",
    )
    return sherpa_onnx.SpokenLanguageIdentification(config)


def swap_tts_model(target_lang):
    """This function is used to swap the tts model real time and freeing the memory occupied by the preloaded tts model if swapped"""
    global current_tts, active_lang

    # if the right model is loaded, then do nothing
    if active_lang == target_lang:
        return current_tts

    print(f"Swapping Model: {active_lang} -> {target_lang}")

    # clear the old model from memory
    if current_tts is not None:
        del current_tts
        gc.collect()  # forcing python to release the memory back to Pi
    
    # load the new model
    if target_lang == "ko":
        current_tts = create_supertonic_tts()
    else:
        current_tts = create_pocket_tts()
    
    active_lang = target_lang
    return current_tts


first_byte_time = 0

# TTS Callbacks

def generated_audio_callback(samples: np.ndarray, progress: float):
    """This function is called when new TTS audio is generated. We put the generated audio into a queue for playback.
    It also detects which model is generating the audio and resamples it on the fly to match the playback thread's 24 kHz requirement"""
    global tts_started, first_byte_time, active_lang, tts_en, tts_ko
    
    if not tts_started:
        first_byte_time = time.time()

    # FIX: if the current language is korean, resample it to match the playback thread
    #      to avoid the deep robotic korean voice
    #if active_lang == "ko":
    #    samples = resample_audio(samples, 16000, 24000)

    current_model = tts_ko if active_lang == "ko" else tts_en
    model_sr = current_model.sample_rate
    target_sr = 24000

    if model_sr != target_sr:
        samples = resample_audio(samples, model_sr, target_sr)

    tts_queue.put(samples)
    tts_started = True
    return 0 if tts_killed else 1


def play_audio_callback(outdata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
    global tts_started, tts_stopped
    if tts_killed or (tts_started and tts_queue.empty() and tts_stopped):
        tts_event.set()

    if tts_queue.empty():
        outdata.fill(0)
        return

    n = 0
    while n < frames and not tts_queue.empty():
        remaining = frames - n
        current_chunk = tts_queue.queue[0]
        k = current_chunk.shape[0]

        if remaining <= k:
            outdata[n:, 0] = current_chunk[:remaining]
            tts_queue.queue[0] = current_chunk[remaining:]
            n = frames
            if tts_queue.queue[0].shape[0] == 0:
                tts_queue.get()
            break
            
        outdata[n : n + k, 0] = tts_queue.get()
        n += k
    
    if n < frames:
        outdata[n:, 0] = 0


def play_audio(sample_rate):
    with sd.OutputStream(
        channels = 1,
        callback = play_audio_callback,
        dtype = "float32",
        samplerate = sample_rate,
        blocksize = 1024,
    ):
        #tts_event.wait()
        while not tts_killed:
            time.sleep(0.1)  # keeping the thread alive
    

def main():
    global tts_started, tts_stopped, tts_event, first_byte_time, active_lang, tts_en, tts_ko, is_speaking

    print("Initializing Baymax Systems...")
    recognizer = create_recognizer()
    denoiser = create_speech_denoiser()
    #slid = whisper_multilingual()
    #tts = create_kokoro_tts()
    tts_en = create_pocket_tts()
    tts_ko = create_supertonic_tts()

    print(f"DEBUG: {tts_en.sample_rate}, {tts_ko.sample_rate}")

    print("Pre-loading reference voice...")
    reference_wav = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
                            
    reference_audio_raw, reference_sample_rate = sf.read(reference_wav)
    if reference_sample_rate != tts_en.sample_rate:
        reference_audio = resample_audio(reference_audio_raw, reference_sample_rate, tts_en.sample_rate)
    else:
        reference_audio = reference_audio_raw.astype(np.float32)

    print("Initialization complete. Starting real-time processing...")

    # OPTIMIZATION: start the playback thread once and leave it open
    # this prevents the driver handshake delay every time bot speaks
    play_back_thread = threading.Thread(target = play_audio, args = (tts_en.sample_rate, ), daemon = True)
    play_back_thread.start()
    
    # VAD Setup
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = VAD_MODEL
    vad_config.silero_vad.min_silence_duration = 0.3
    vad_config.sample_rate = 16000

    window_size = vad_config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds = 10)

    # Audio IO Setup
    mic_sample_rate = 48000
    samples_per_read = int(0.1 * mic_sample_rate)  # read 100ms of audio at a time

    buffer = np.array([], dtype = np.float32)
    #texts = []

    with sd.InputStream(channels = 1, dtype = "float32", samplerate = mic_sample_rate) as s:
        while True:
            samples, overflowed = s.read(samples_per_read)
            
            if is_speaking:
                # if baymax is speaking we discard the audio immediately
                # to prevent processing any echo
                continue
            
            if overflowed:
                # if the pi was too slow, the mic buffer will overflow
                # we must skip this to catch up
                continue
            
            samples = samples.reshape(-1)
            samples = resample_audio(samples, mic_sample_rate, 16000)
            #samples = samples.flatten()[::3]
            # frame_shift = denoiser.frame_shift_in_samples
            # Denoised = []

            # for start in range(0, len(samples), frame_shift):
            #     chunk = samples[start : start + frame_shift]
            #     denoised = denoiser(chunk, 16000)
            #     Denoised.append(np.array(denoised.samples, dtype = np.float32))

            # Denoised.append(np.asarray(denoiser.flush().samples, dtype = np.float32))

            # add to VAD
            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            if not vad.empty():
                # drain the VAD queue and keep the last one
                latest_segment = None
                while not vad.empty():
                    latest_segment = vad.front
                    vad.pop()

                if latest_segment and len(latest_segment.samples) > 8000:
                    raw_speech = latest_segment.samples
                    clean_speech = denoiser(raw_speech, 16000).samples
                    #clean_speech = raw_speech

                    # Transcribe
                    asr_stream = recognizer.create_stream()
                    asr_stream.accept_waveform(16000, clean_speech)
                    recognizer.decode_stream(asr_stream)

                    # Identify Language
                    # slid_stream = slid.create_stream()
                    # slid_stream.accept_waveform(sample_rate = 16000, waveform = clean_speech)
                    # lang = slid.compute(slid_stream)

                    text = asr_stream.result.text.strip().lower()
                    if len(text) > 1:
                        #idx = len(texts)
                        #texts.append(text)
                    
                        # identifying the spoken language
                        if bool(re.search('[가-힣]', text)):
                            lang = "ko"
                        elif bool(re.search('[A-Za-z]', text)):
                            lang = "en"
                        else:
                            continue
                        active_lang = lang

                        print(f"User: {text}")
                        print(lang)
                        print("Thinking...")

                        is_speaking = True  # set the speaker state

                        while not tts_queue.empty(): tts_queue.get()
                        tts_event.clear()
                        tts_started = False
                        tts_stopped = False
                        start_trigger = time.time()
                        
                        # OPTIMIZATION: We run generation in a thread (non-blocking)
                        def run_tts(tts_text, target_lang):
                            """ This function triggers audio callback"""
                            global tts_stopped
                            
                            # swapping the model only if necessary
                            #tts_engine = swap_tts_model(target_lang)

                            gen_config = sherpa_onnx.GenerationConfig()
                            
                            if target_lang == "ko":
                                gen_config.sid = 1   # female speaker
                                gen_config.num_steps = 12
                                gen_config.extra["lang"] = "ko"
                                tts_ko.generate(tts_text, gen_config, callback = generated_audio_callback)
                            else:
                                gen_config.reference_audio = reference_audio # Required for Pocket-TTS
                                gen_config.reference_sample_rate = tts_en.sample_rate #reference_sample_rate
                                gen_config.num_steps = 5
                                tts_en.generate(text, gen_config, callback = generated_audio_callback)
                            
                            tts_stopped = True # signaling the callback that generation is done
                        
                        threading.Thread(target = run_tts, args = (text, lang), daemon = True).start()
                        
                        while not tts_started and not tts_stopped:
                            time.sleep(0.01)

                        # Statistics for tuning
                        print(f"Time to FIRST sound: {first_byte_time - start_trigger}")
                        #print(f"Full sentence generation: {full_gen_end - start_trigger}")

                        # wait for baymax to finish speaking before listening again
                        # this prevents the mic from picking up the speakers
                        tts_event.wait()

                        # reset state
                        is_speaking = False

                        buffer = np.array([], dtype = np.float32)


if __name__ == "__main__":
    main()