#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <fftw3.h>

class CKokoro {
    public:
        CKokoro(const std::string& model_file, const std::string& config, const std::string& voice_path);
        ~CKokoro();

    private:
        std::string trim(const std::string& s);
        int split(const std::string& text, 
            std::vector<std::string>& phrases, 
            std::vector<std::string>& punctuations);
        std::string post_process_phonemes(const char * phonemes);
        int get_utf8_char(const std::string text, std::vector<std::string>& chars);
        int load_vocab(const std::string& config);
        int load_voices(const std::string& voice_path);
        int pre_process(const std::string& text, std::vector<int64_t>& input_ids);
        int post_process(std::vector<float>& audio);

    public:
        int tts(const std::string& text, const std::string& style, std::vector<float>& audio);

    private:
        std::map<std::string, int> vocab;
        std::map<std::string, nlohmann::json> voices;
        std::unique_ptr<Ort::Session> session;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

        const int frame_size = 2048;
        const int hop_size = 512;
        const int sample_rate = 24000;
        const float cutoff_freq = 4500;
        const float attenuation = 0.1;
        const float amplification = 2.0;
        const int cutoff_bin = static_cast<int>(cutoff_freq * frame_size / sample_rate);

        fftwf_plan fft_plan;
        fftwf_plan ifft_plan;
        std::vector<float> stft_window = std::vector<float>(frame_size);

        const char * file_dict = "data/jieba.dict.utf8";
        const char * file_hmm = "data/hmm_model.utf8";
        const char * file_user_dict = "data/user.dict.utf8";
        const char * file_idf = "data/idf.utf8";
        const char * file_stop_word = "data/stop_words.utf8";
};