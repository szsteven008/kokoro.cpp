#include <boost/program_options.hpp>
#include <cstring>
#include <string>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <portaudio.h>
#include <sndfile.h>

#include "kokoro.h"

typedef struct _tagPlayState {
    float * data;
    int total;
    int next;
} PlayState;

const int channelCount = 1;
const double sampleRate = 24000;
const unsigned long framesPerBuffer = 128;

int stream_callback(
    const void *input, 
    void *output,
    unsigned long frameCount,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData ) {
    (void)input;
    (void)timeInfo;
    (void)statusFlags;

    PlayState * state = (PlayState *)userData;
    if (state->next >= state->total) return paComplete;

    float * p = reinterpret_cast<float *>(output);
    for (int i=0; i<frameCount; ++i) {
        p[i] = *(state->data + state->next + i);
    }
    state->next += frameCount;

    return paContinue;
}

int play(const std::vector<float>& audio) {
    PlayState play_state {
        const_cast<float *>(audio.data()), static_cast<int>(audio.size()), 0
    };

    PaError ret;
    ret = Pa_Initialize();
    if (ret != paNoError) return -1;

    PaStreamParameters parameters;
    parameters.device = Pa_GetDefaultOutputDevice();
    parameters.channelCount = channelCount;
    parameters.sampleFormat = paFloat32;
    parameters.suggestedLatency = Pa_GetDeviceInfo(parameters.device)->defaultLowOutputLatency;
    parameters.hostApiSpecificStreamInfo = nullptr;

    PaStream * stream = nullptr;
    ret = Pa_OpenStream(&stream, 
                        nullptr, 
                        &parameters, 
                        sampleRate, 
                        framesPerBuffer, 
                        0, 
                        stream_callback, 
                        &play_state);
    if (ret != paNoError) goto play_error;
    ret = Pa_StartStream(stream);
    while (Pa_IsStreamActive(stream) > 0) {
        usleep(1000);
    }
    Pa_StopStream(stream);
    Pa_CloseStream(stream);

play_error:
    Pa_Terminate();
    return 0;
}

int save(const std::vector<float>& audio) {
    SF_INFO info;
    memset(&info, 0, sizeof(info));
    info.channels = 1;
    info.samplerate = 24000;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE * f = sf_open("output.wav", SFM_WRITE, &info);
    sf_write_float(f, audio.data(), audio.size());
    sf_close(f);

    return 0;
}

int main(int argc, char * argv[]){
    boost::program_options::options_description opts("Allowed options");
    opts.add_options()
                    ("help,h", "show help message.")
                    ("model,m", 
                        boost::program_options::value<std::string>()->default_value("models/kokoro.onnx"), 
                        "model file.")
                    ("vocab,c", 
                        boost::program_options::value<std::string>()->default_value("models/vocab.json"), 
                        "vocab file.")
                    ("voices,v", 
                        boost::program_options::value<std::string>()->default_value("voices"), 
                        "voices path")
                    ("text,t", boost::program_options::value<std::string>(), "text for speaking")
                    ("style,s", 
                        boost::program_options::value<std::string>()->default_value("af_heart"), 
                        "style for speaking")
                    ;
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, opts), 
        vm); 
    if (vm.count("help") > 0 || !vm.count("text")) {
        std::cout << opts << std::endl;
        return -1;
    }

    std::string model = vm["model"].as<std::string>();
    std::string config = vm["vocab"].as<std::string>();
    std::string voices = vm["voices"].as<std::string>();
    std::string text = vm["text"].as<std::string>();
    std::string style = vm["style"].as<std::string>();

    CKokoro kokoro(model, config, voices);
    
    std::vector<float> audio;
    kokoro.tts(text, style, audio);

    std::cout << "audio: " << audio.size() << std::endl;

    save(audio);
    play(audio);

    std::cout << argv[0] << " ok!" << std::endl;

    return 0;
}