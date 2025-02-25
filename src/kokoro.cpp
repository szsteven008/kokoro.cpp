#include "kokoro.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <nlohmann/json_fwd.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include <espeak-ng/speak_lib.h>
#include <iostream>

CKokoro::CKokoro(const std::string& model_file, const std::string& config, const std::string& voice_path) {
    load_vocab(config);
    load_voices(voice_path);

    espeak_Initialize(
        AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);

    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "kokoro");

    Ort::SessionOptions so;
    session = std::make_unique<Ort::Session>(env, model_file.c_str(), so);

    fft_plan = fftwf_plan_dft_r2c_1d(frame_size, nullptr, nullptr, FFTW_ESTIMATE);
    ifft_plan = fftwf_plan_dft_c2r_1d(frame_size, nullptr, nullptr, FFTW_ESTIMATE);
    for (int i=0; i<stft_window.size(); ++i) {
        stft_window[i] = 0.54f - 0.46f * std::cos(2 * M_PI * i / (frame_size - 1));
    }
}

CKokoro::~CKokoro() {
    fftwf_destroy_plan(ifft_plan);
    fftwf_destroy_plan(fft_plan);
    session->release();
    espeak_Terminate();
}

std::string CKokoro::trim(const std::string& s) {
    auto begin = 
        std::find_if_not(s.begin(), s.end(), ::isspace);
    auto end = 
        std::find_if_not(s.rbegin(), s.rend(), ::isspace).base();
    return (begin < end) ? std::string(begin, end) : "";
}

int CKokoro::split(const std::string& text, 
    std::vector<std::string>& phrases, 
    std::vector<std::string>& punctuations) {
    std::regex re(R"([;:,.!?¡¿—…"«»“”\(\)\{\}\[\]])");
    auto begin = 
        std::sregex_token_iterator(text.begin(), text.end(), re, {-1, 0});
    auto end = std::sregex_token_iterator();
    for (auto it = begin; it != end; it++) {
        std::string sub_text = trim(it->str());
        std::string punctuation = "";
        if (strlen(it->second.base())) { 
            it++;
            punctuation = it->str();
        }

        if (sub_text.size()) {
            phrases.emplace_back(sub_text);
            if (punctuation.size()) {
                punctuations.emplace_back(punctuation);
            }
        }
    }
    return 0;
}

std::string CKokoro::post_process_phonemes(const char * phonemes) {
    const std::map<std::string, std::string> m = {
        { R"(ʔˌn\u0329)", "tn" }, 
        { R"(ʔn\u0329)", "tn" }, 
        { R"(ʔn)", "tn" }, 
        { R"(ʔ)", "t" }, 
        { R"(aɪ)", "I" }, 
        { R"(aʊ)", "W" }, 
        { R"(dʒ)", "ʤ" }, 
        { R"(eɪ)", "A" }, 
        { R"(e)", "A" }, 
        { R"(tʃ)", "ʧ" }, 
        { R"(ɔɪ)", "Y" }, 
        { R"(əl)", "ᵊl" }, 
        { R"(ʲo)", "jo" }, 
        { R"(ʲə)", "jə" }, 
        { R"(ʲ)", "" }, 
        { R"(ɚ)", "əɹ" }, 
        { R"(r)", "ɹ" }, 
        { R"(x)", "k" }, 
        { R"(ç)", "k" }, 
        { R"(ɐ)", "ə" }, 
        { R"(ɬ)", "l" }, 
        { R"(\u0303)", "" }, 
        { R"(oʊ)", "O" }, 
        { R"(ɜːɹ)", "ɜɹ" }, 
        { R"(ɜː)", "ɜɹ" }, 
        { R"(ɪə)", "iə" }, 
        { R"(ː)", "" } 
    };
    std::string s(phonemes);
    for (auto& [key, value]: m) {
        s = std::regex_replace(s, std::regex(key), value);
    }
    s = std::regex_replace(s, std::regex(R"(_)"), "");
    return s;
}

int CKokoro::get_utf8_char(const std::string text, std::vector<std::string>& chars) {
    for (int i=0; i<text.size();) {
        int len = 1;
        if ((text[i] & 0xe0) == 0xc0) {
            len = 2;
        } else if ((text[i] & 0xf0) == 0xe0) {
            len = 3;
        } else if ((text[i] & 0xf8) == 0xf0) {
            len = 4;
        }

        chars.emplace_back(text.substr(i, len));
        i += len;
    }
    return 0;
}

int CKokoro::load_vocab(const std::string& config) {
    std::ifstream f(config.c_str());
    nlohmann::json j;
    f >> j;
    vocab = j.get<std::map<std::string, int>>();

    return 0;
}

int CKokoro::load_voices(const std::string& voice_path) {
    std::filesystem::path path(voice_path);
    for (auto& dir_entry: std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_directory(dir_entry.path())) continue;
        std::string name = dir_entry.path().stem();

        std::ifstream f(dir_entry.path().c_str());
        nlohmann::json value;
        f >> value;

        voices[name] = value;
    }

    return 0;
}

int CKokoro::pre_process(const std::string& text, std::vector<int64_t>& input_ids) {
    std::string s = text;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    std::vector<std::string> phrases;
    std::vector<std::string> punctuations;
    split(s, phrases, punctuations);

    espeak_VOICE voice;
    memset(&voice, 0, sizeof(voice));
    voice.name = "English_(America)";
    voice.languages = "en-us";
    voice.gender = 2;
    espeak_SetVoiceByProperties(&voice);

    input_ids = { 0 };
    int phonememode = ('_' << 8) | 0x02;
    for (int i=0; i<phrases.size(); ++i) {
        std::string phrase = phrases[i];
        if (phrase.size()) {
            const char * p = phrase.c_str();
            const char * phonemes = espeak_TextToPhonemes(
                reinterpret_cast<const void **>(&p), espeakCHARS_UTF8, phonememode);
    
            std::string s = post_process_phonemes(phonemes);
            std::vector<std::string> chars;
            get_utf8_char(s, chars);
            for (auto& c: chars) {
                input_ids.emplace_back(vocab[c]);
            }    
        }
        if (punctuations.size() > i) {
            std::string punctuation = punctuations[i];
            input_ids.emplace_back(vocab[punctuation]);
            input_ids.emplace_back(vocab[" "]);
        }
    }
    input_ids.emplace_back(0);

    return 0;
}

int CKokoro::post_process(std::vector<float>& audio) {
    int num_frames = (audio.size() - frame_size) / hop_size + 1;
    std::vector<std::vector<float>> frames(num_frames, std::vector<float>(frame_size));
    float * fft_in = fftwf_alloc_real(frame_size);
    fftwf_complex * fft_out = fftwf_alloc_complex(frame_size / 2 + 1);
    float * ifft_out = fftwf_alloc_real(frame_size);
    for (int n=0; n<num_frames; ++n) {
        const int start = n * hop_size;
        for (int i=0; i<frame_size; ++i) {
            const int idx = start + i;
            fft_in[i] = (idx < audio.size()) ? audio[idx] * stft_window[i] : 0.0;
        }

        fftwf_execute_dft_r2c(fft_plan, fft_in, fft_out);
        for (int k=0; k<(frame_size/2+1); ++k) {
            if (k >= cutoff_bin) {
                fft_out[k][0] *= attenuation;
                fft_out[k][1] *= attenuation;    
            } else {
                fft_out[k][0] *= amplification;
                fft_out[k][1] *= amplification;    
            }
        }

        fftwf_execute_dft_c2r(ifft_plan, fft_out, ifft_out);
        for (int i=0; i<frame_size; ++i) {
            frames[n][i] = ifft_out[i] / frame_size * stft_window[i];
        }
    }

    std::vector<float> output(audio.size(), 0.0);
    for (int n=0; n<num_frames; ++n) {
        const int start = n * hop_size;
        for (int i=0; i<frame_size; ++i) {
            if ((start + i) < audio.size()) {
                output[start + i] += frames[n][i];
            }
        }
    }

    fftwf_free(ifft_out);
    fftwf_free(fft_out);
    fftwf_free(fft_in);

    audio = output;

    return 0;
}

int CKokoro::tts(const std::string& text, const std::string& style, std::vector<float>& audio) {
    std::vector<int64_t> input_ids;
    pre_process(text, input_ids);
    std::cout << "text: " << text << " -> " << "input_ids: ";
    for (auto& item: input_ids) std::cout << item << ", ";
    std::cout << std::endl;

    std::vector<float> voice = voices[style][input_ids.size()-2].get<std::vector<float>>();
    std::vector<int> speed = { 1 };

    std::vector<Ort::Value> inputs;
    std::vector<int64_t> shape_input_ids = { 1, static_cast<int64_t>(input_ids.size()) };
    inputs.emplace_back(
        Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), 
            shape_input_ids.data(), shape_input_ids.size()));
    std::vector<int64_t> shape_style = { 1, 256 };
    inputs.emplace_back(
        Ort::Value::CreateTensor<float>(
            memory_info, voice.data(), voice.size(), 
            shape_style.data(), shape_style.size()));
    std::vector<int64_t> shape_speed = { 1 };
    inputs.emplace_back(
        Ort::Value::CreateTensor<int>(
            memory_info, speed.data(), speed.size(), 
            shape_speed.data(), shape_speed.size()));

    std::vector<const char *> input_names = { "input_ids", "style", "speed" };
    std::vector<const char *> output_names = { "waveform" };
    
    Ort::RunOptions options;
    auto outputs = session->Run(options, 
                                                    input_names.data(), 
                                                    inputs.data(), 
                                                    inputs.size(), 
                                                    output_names.data(), 
                                                    output_names.size());
    if (!outputs.size()) return -1;
    std::vector<int64_t> shape_audio = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int len = shape_audio[1];
    audio.resize(len);
    memcpy(audio.data(), outputs[0].GetTensorData<float *>(), len * sizeof(float));

    //post_process(audio);
    return 0;
}
