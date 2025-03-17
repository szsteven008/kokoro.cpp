set(FILES main.cpp kokoro.cpp)

add_executable(main ${FILES})
set_target_properties(main PROPERTIES BUILD_RPATH "/usr/local/lib")
target_link_libraries(main 
                      Boost::program_options 
                      onnxruntime 
                      portaudio 
                      sndfile 
                      fftw3f 
                      espeak-ng)