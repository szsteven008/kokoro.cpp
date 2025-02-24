set(FILES main.cpp kokoro.cpp)

add_executable(main ${FILES})
target_link_libraries(main 
                      Boost::program_options 
                      onnxruntime 
                      portaudio 
                      sndfile 
                      fftw3f 
                      espeak-ng)