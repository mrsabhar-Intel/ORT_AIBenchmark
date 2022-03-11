# ORT_AIBenchmark
Simple benchmark to evaluate execution providers 

ORT package compilation 
-build.bat --config Release --use_cuda --cudnn_home cudnn_package --cuda_home  cuda_package --cuda_version 11.5 --use_openvino GPU_FP16_NO_PARTITION --use_dml --build_shared_lib

ORT  -- 
    ONNX Runtime - https://github.com/Microsoft/onnxruntime
    OpenVINO version - https://github.com/openvinotoolkit/openvino
    Install NVIDIA(R) CUDA based on documentation from onnxruntime - https://onnxruntime.ai/docs/build/eps.html#cuda
    Intel OpenVINO - https://onnxruntime.ai/docs/build/eps.html#openvino


Linkers - 
onnxruntime.lib
DirectML.lib
onnxruntime_providers_dml.lib
C:\Intel\openvino_2021.4.752\opencv\lib\opencv_core453.lib
C:\Intel\openvino_2021.4.752\opencv\lib\opencv_dnn453.lib
C:\Intel\openvino_2021.4.752\opencv\lib\opencv_imgcodecs453.lib
C:\Intel\openvino_2021.4.752\opencv\lib\opencv_imgproc453.lib
