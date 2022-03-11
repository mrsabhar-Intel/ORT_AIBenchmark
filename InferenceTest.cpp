

#include <dml/dml_provider_factory.h>
#include <onnxruntime_cxx_api.h>
//#include <cuda/cuda_provider_options.h>
#include <direct.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>  // To use runtime_error
#include <string>
#include <vector>

OrtCUDAProviderOptions CreateDefaultOrtCudaProviderOptionsWithCustomStream(void* cuda_compute_stream) {
  OrtCUDAProviderOptions cuda_options;

  cuda_options.device_id = 0;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
  cuda_options.arena_extend_strategy = 0;
  cuda_options.do_copy_in_default_stream = true;
  cuda_options.has_user_compute_stream = cuda_compute_stream != nullptr ? 1 : 0;
  cuda_options.user_compute_stream = cuda_compute_stream;
  cuda_options.default_memory_arena_cfg = nullptr;

  return cuda_options;
}
template <typename T>
T vectorProduct(const std::vector<T>& v) {
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

// Function to validate the input model file extension.
bool checkModelExtension(const std::string& filename) {
  if (filename.empty()) {
    throw std::runtime_error("[ ERROR ] The Model file path is empty");
  }
  size_t pos = filename.rfind('.');
  if (pos == std::string::npos) return false;
  std::string ext = filename.substr(pos + 1);
  if (ext == "onnx") return true;
  return false;
}

// Handling divide by zero
float division(float num, float den) {
  if (den == 0) {
    throw std::runtime_error("[ ERROR ] Math error: Attempted to divide by Zero\n");
  }
  return (num / den);
}
// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

void printHelp() {
  std::cout << "To run the model, use the following command:\n";
  std::cout << "Example: InterenceTest --use_openvino <path_to_the_model> " << std::endl;
  std::cout << "\n To Run using OpenVINO EP.\nExample: InterenceTest --use_openvino modelname.onnx \n" << std::endl;
  std::cout << "\n To Run on DirectML GPU through ORT.\n Example: InterenceTest --use_dml modelname.onnx \n"
            << std::endl;
  std::cout << "\n To Run on WinML DirectML.\n Example: run_inference --use_cpu modelname.onnx \n" << std::endl;
}

int main(int argc, char* argv[]) {
  bool useOPENVINO{false};
  bool useDML{false};
  bool useCUDA{false};
  const char* useOPENVINOFlag = "--use_openvino";
  const char* useDMLFlag = "--use_dml";
  const char* useCUDAFlag = "--use_cuda";
  extern std::unique_ptr<Ort::Env> ort_env;
  if (argc == 2) {
    std::string option = argv[1];
    if (option == "--help" || option == "-help" || option == "--h" || option == "-h") {
      printHelp();
    }
    return 0;

  } else if (strcmp(argv[1], useOPENVINOFlag) == 0) {
    useOPENVINO = true;
  } else if (strcmp(argv[1], useDMLFlag) == 0) {
    useDML = true;
  } else if (strcmp(argv[1], useCUDAFlag) == 0) {
    useCUDA = true;
  }

  if (useOPENVINO) {
    std::cout << "Inference Execution Provider: ORT + OPENVINO" << std::endl;
  } else if (useDML) {
    std::cout << "Inference Execution Provider: ORT + DirectML" << std::endl;
  } else if (useCUDA) {
    std::cout << "Inference Execution Provider: ORT + CUDA " << std::endl;
  }
#ifdef _WIN32
  std::string str = argv[2];
  std::wstring wide_string = std::wstring(str.begin(), str.end());
  std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
  std::string model_file = argv[1];
#endif

  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Inference_Benchmark");
  Ort::SessionOptions sessionOptions;
  // sessionOptions.SetIntraOpNumThreads(1);

  //// Appending OpenVINO Execution Provider API
  if (useOPENVINO) {
    // Using OPENVINO backend
    OrtOpenVINOProviderOptions options;
    bool use_compiled_network = true;
    options.device_type = "GPU_FP16";  // Other options are: GPU_FP32 or  GPU_FP16 . Default FP16
    std::string blob_dump_path = "cache";

    if (_mkdir(blob_dump_path.c_str()) == 0) {
      printf("Cache directory was successfully created\n");
    } else
      printf("Problem creating directory cache directory\n");
    options.blob_dump_path = blob_dump_path.c_str();
    options.use_compiled_network = use_compiled_network;
    std::cout << "OpenVINO device type is set to: " << options.device_type << std::endl;
    sessionOptions.AppendExecutionProvider_OpenVINO(options);

  } else if (useDML) {
    // Use DML Backend
    sessionOptions.DisableMemPattern();
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));

  }
   else if (useCUDA) {
      //Use CUDA backend
      OrtCUDAProviderOptions options;
      options.device_id = 0;
      options.arena_extend_strategy = 0;
      options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
      options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
      options.do_copy_in_default_stream = 1;
      sessionOptions.AppendExecutionProvider_CUDA(options);
   
  }
  else {
    // fallback to CPU
    std::cout << "CPU Version" << std::endl;
  }

  Ort::Session session(nullptr);
  std::chrono::steady_clock::time_point begin_session = std::chrono::steady_clock::now();
  try {
    Ort::Session session_(env, model_file.c_str(), sessionOptions);
    session = std::move(session_);
  } catch (const Ort::Exception& ex) {
    std::cout << "Fallback to CPU Version" << std::endl;
    Ort::SessionOptions sessionOptions_;
    Ort::Session session_cpu(env, model_file.c_str(), sessionOptions_);
    session = std::move(session_cpu);
  }
  std::chrono::steady_clock::time_point end_session = std::chrono::steady_clock::now();
  std::cout << "Session Creation time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_session - begin_session).count() << " ms"
            << std::endl;

  size_t numInputNodes = session.GetInputCount();
  size_t numOutputNodes = session.GetOutputCount();
  std::vector<const char*> inputName;
  std::vector<Ort::TypeInfo> inputTypeInfo;
  std::vector<Ort::TypeInfo> outputTypeInfo;
  std::vector<const char*> outputName;
  std::vector<const char*> inputNames;
  std::vector<const char*> outputNames;
  std::vector<Ort::Value> outputTensors;
  std::vector<Ort::Value> inputTensors;
  std::vector<size_t> inputTensorSize;
  std::vector<std::vector<int64_t>> xArray;
  std::vector<std::vector<float>> inputTensorValues;

  std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
  std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
  Ort::AllocatorWithDefaultOptions allocator;
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  for (int i = 0; i < numInputNodes; i++) {
    inputName.push_back(session.GetInputName(i, allocator));
    std::cout << "Input Name: " << inputName[0] << std::endl;
    inputTypeInfo.push_back(session.GetInputTypeInfo(i));
    std::vector<int64_t> inputDims = inputTypeInfo[i].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;
    // free dimensions are treated as 1 if not overriden
    for (int64_t& dim : inputDims) {
      if (dim == -1) {
        dim = 1;
      }
    }
    std::cout << "Input Dimensions after : " << inputDims << std::endl;
    xArray.push_back(inputDims);

    inputTensorSize.push_back(vectorProduct(xArray[i]));
    std::vector<float> inputTensorValue(inputTensorSize[i]);
    inputTensorValues.push_back(inputTensorValue);
    std::generate(inputTensorValues[i].begin(), inputTensorValues[i].end(),
                  [&] { return rand() % 255; });  // generate random numbers in the range [0, 255]
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues[i].data(), inputTensorSize[i],
                                                           xArray[i].data(), xArray[i].size()));
  }

  for (int out = 0; out < numOutputNodes; out++) {
    outputName.push_back(session.GetOutputName(out, allocator));
    std::cout << "Output Name: " << outputName << std::endl;
    outputTypeInfo.push_back(session.GetOutputTypeInfo(out));

    std::vector<int64_t> outputDims = outputTypeInfo[out].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;
    size_t outputTensorSize = vectorProduct(outputDims);
  }

  // auto output_tensors;
  std::chrono::steady_clock::time_point begin_fil = std::chrono::steady_clock::now();
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputName.data(), inputTensors.data(),
                                                    inputName.size(), outputName.data(), outputName.size());
  std::chrono::steady_clock::time_point end_fil = std::chrono::steady_clock::now();
  std::cout << "1st Inference Latency: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_fil - begin_fil).count() << " ms" << std::endl;

  // Measure latency
  int numTests{100};
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // Run: Running the session is done in the Run() method:
  for (int i = 0; i < numTests; i++) {
    session.Run(Ort::RunOptions{nullptr}, inputName.data(), inputTensors.data(), inputName.size(), outputName.data(),
                outputName.size());
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Minimum Inference Latency: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / static_cast<float>(numTests)
            << " ms" << std::endl;

  return 0;
}
