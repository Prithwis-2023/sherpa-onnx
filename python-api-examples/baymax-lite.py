import threading
import queue
import time
import numpy as np
import sherpa_onnx
import sounddevice as sd
import soundfile as sf
import re
import gc  # for manual memory cleaning
import json
import os
import argparse
from functools import partial

# global state for tts playback
tts_queue = queue.Queue()
tts_event = threading.Event()
tts_started = False  # set to true once generated_audio_callback is called
tts_stopped = False  # set to true once all text is processed
tts_killed = False   # set to true once exited

# global state for the current active tts
current_tts = None
active_lang = None  # 'en' or 'ko'

tts = None

is_speaking = False  # global flag to prevent Baymax from hearing its own echo
baymax_asks = True

first_byte_time = 0

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--lang",
    type = str,
    default = "en",
    help = "Language mode: either en or ko."
)
args = parser.parse_args()

if args.lang == "en":
    active_lang = "en"
else:
    active_lang = "ko"

# configuration
# wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
VAD_MODEL = "./silero_vad.onnx"

#curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
#tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
#rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
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

#wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
#tar xf vits-piper-en_US-amy-low.tar.bz2
VITS_PIPER_MODEL = "./vits-piper-en_US-amy-low/en_US-amy-low.onnx"
VITS_PIPER_LEXICON = ""
VITS_PIPER_TOKENS = "./vits-piper-en_US-amy-low/tokens.txt"
VITS_PIPER_DATA_DIR = "./vits-piper-en_US-amy-low/espeak-ng-data"


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

def create_recognizer(language):
    """Setup for ASR. We are using SenseVoice"""
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model = ASR_MODEL,
        tokens = ASR_TOKENS,
        num_threads = 3,
        use_itn = True,
        language = language, # one out of "zh", "en", "ja", "ko", "yue"
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


def create_vits_tts():
    config = sherpa_onnx.OfflineTtsConfig(
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model = VITS_PIPER_MODEL,
                data_dir = VITS_PIPER_DATA_DIR,
                tokens = VITS_PIPER_TOKENS,
            ), 
            provider = "cpu",
            debug = False,
            num_threads = 4,
        ),
        rule_fsts = "",
        max_num_sentences = 1
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


# TTS Callbacks

def generated_audio_callback(samples: np.ndarray, progress: float, target_sr: int):
    """This function is called when new TTS audio is generated. We put the generated audio into a queue for playback.
    It also detects which model is generating the audio and resamples it on the fly to match the playback thread's 24 kHz requirement"""
    global tts_started, first_byte_time, active_lang, tts
    
    if not tts_started:
        first_byte_time = time.time()

    # FIX: if the current language is korean, resample it to match the playback thread
    #      to avoid the deep robotic korean voice
    #if active_lang == "ko":
    #    samples = resample_audio(samples, 16000, 24000)

    current_model = tts
    model_sr = current_model.sample_rate
    #target_sr = 24000

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
    

def baymax_say(text, reference_audio, stream):
    """This function consists of the regex spoken language identification and tts component"""
    global active_lang, is_speaking, tts_queue, tts_event, tts_started, tts_stopped, tts_en, tts_ko

    # # identifying the spoken language
    # if bool(re.search('[가-힣]', text)):
    #     lang = "ko"
    # elif bool(re.search('[A-Za-z]', text)):
    #     lang = "en"
    # else:
    #     lang = "en"  # default to English if no language detected

    # active_lang = lang

    is_speaking = True  # set the speaker state

    stream.stop() # pause the microphone capture

    try:
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
                ko_callback = partial(generated_audio_callback, target_sr=24000)
                tts.generate(tts_text, gen_config, callback = ko_callback)
            else:
                # gen_config.reference_audio = reference_audio # Required for Pocket-TTS
                # gen_config.reference_sample_rate = tts.sample_rate #reference_sample_rate
                # gen_config.num_steps = 5
                gen_config.sid = 0
                gen_config.speed = 1.0
                gen_config.silence_scale = 0.2
                en_callback = partial(generated_audio_callback, target_sr=16000)
                tts.generate(text, gen_config, callback = en_callback)
            
            tts_stopped = True # signaling the callback that generation is done
        
        threading.Thread(target = run_tts, args = (text, active_lang), daemon = True).start()
        
        while not tts_started and not tts_stopped:
            time.sleep(0.01)

        # Statistics for tuning
        print(f"Time to FIRST sound: {first_byte_time - start_trigger}")
        #print(f"Full sentence generation: {full_gen_end - start_trigger}")

        # wait for baymax to finish speaking before listening again
        # this prevents the mic from picking up the speakers
        tts_event.wait()

    finally:
        stream.start()  # resume microphone capture

    # reset state
    is_speaking = False


def load_questions(filepath):
    """This function loads the quiz questions json file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print("File not found. Exiting...")
        return


def main():
    global tts_started, tts_stopped, tts_event, first_byte_time, active_lang, tts, is_speaking, baymax_asks

    print("Initializing Baymax Systems...")

    denoiser = create_speech_denoiser()
    #slid = whisper_multilingual()
    #tts = create_kokoro_tts()

    if active_lang == "en":
        recognizer = create_recognizer("en")
        tts = create_vits_tts()
        QUIZ_QUESTIONS = load_questions("./questions_en.json")
    else:
        recognizer = create_recognizer("ko")
        tts = create_supertonic_tts()
        QUIZ_QUESTIONS = load_questions("./questions_ko.json")

    #tts_en = create_pocket_tts()
    #tts_ko = create_supertonic_tts()

    #recognizer_en = create_recognizer("en")
    #recognizer_ko = create_recognizer("ko")

    reference_wav = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav"
    
    #QUIZ_QUESTIONS = load_questions("./questions.json")
    current_q_id = 0

    print(f"DEBUG: {tts.sample_rate}")

    print("Pre-loading reference voice...")
    reference_audio_raw, reference_sample_rate = sf.read(reference_wav)
    if reference_sample_rate != tts.sample_rate:
        reference_audio = resample_audio(reference_audio_raw, reference_sample_rate, tts.sample_rate)
    else:
        reference_audio = reference_audio_raw.astype(np.float32)

    print("Initialization complete. Starting real-time processing...")

    # OPTIMIZATION: start the playback thread once and leave it open
    # this prevents the driver handshake delay every time bot speaks
    play_back_thread = threading.Thread(target = play_audio, args = (tts.sample_rate, ), daemon = True)
    #play_back_thread = threading.Thread(target = play_audio, args = (24000, ), daemon = True)
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
        while baymax_asks:
            # Ask the question once
            question = QUIZ_QUESTIONS[current_q_id]["question"]
            baymax_say(question, reference_audio, s)
            
            #buffer = np.array([], dtype = np.float32)
            time.sleep(0.5)  # Wait for audio to fully finish playing
            
            # recognizer = recognizer_en if active_lang == "en" else recognizer_ko
            print("Listening for answer...")
            answer_received = False
            
            # Listen for answer until received
            while not answer_received:
                samples, overflowed = s.read(samples_per_read)

                if overflowed:
                    continue
                
                samples = samples.reshape(-1)
                samples = resample_audio(samples, mic_sample_rate, 16000)

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

                        # Transcribe
                        asr_stream = recognizer.create_stream()
                        asr_stream.accept_waveform(16000, clean_speech)
                        recognizer.decode_stream(asr_stream)

                        reply = asr_stream.result.text.strip().lower()

                        if len(reply) >= 1:
                            print(f"User: {reply}")
                            
                            if QUIZ_QUESTIONS[current_q_id]["answer"] in reply:
                                baymax_say(f"That's correct! Moving on.", reference_audio, s) if active_lang == "en" else baymax_say("맞습니다!", reference_audio, s)
                                current_q_id += 1
                                answer_received = True
                                #buffer = np.array([], dtype = np.float32)
                                time.sleep(0.2)
                            else:
                                baymax_say(f"Incorrect! Here is a hint: {QUIZ_QUESTIONS[current_q_id]['hint']}", reference_audio, s) if active_lang == "en" else baymax_say(f"틀렸습니다! 힌트를 드리겠습니다: {QUIZ_QUESTIONS[current_q_id]['hint']}", reference_audio, s)
                                #buffer = np.array([], dtype = np.float32)
                                time.sleep(0.2)
            
                buffer = np.array([], dtype = np.float32)

            if current_q_id >= len(QUIZ_QUESTIONS):
                baymax_asks = False
            

        
if __name__ == "__main__":
    main()