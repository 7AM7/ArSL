#include "mediapipe/calculators/arsign/sign_lang_prediction_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/status.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <math.h>
#include <chrono>
#include <future>
#include <chrono>
#include <thread>

using namespace mediapipe;

namespace signlang
{
    constexpr char kLandmarksTag[] = "LANDMARKS";
    constexpr char kTextOutputTag[] = "TEXT";
    constexpr float defaultPoint = 0.0F;
    // Example config:
    // node {
    //   calculator: "SignLangPredictionCalculator"
    //   input_stream: "LANDMARKS:landmarks"
    //   output_stream: "TEXT:prediction"
    // }
    class SignLangPredictionCalculator : public CalculatorBase
    {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract *cc);
        ::mediapipe::Status Open(CalculatorContext *cc) override;
        ::mediapipe::Status Process(CalculatorContext *cc) override;

    private:
        ::mediapipe::Status LoadOptions(CalculatorContext *cc);
        void AddHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc);
        ::mediapipe::Status UpdateFrames(CalculatorContext *cc);
        bool ShouldPredict();
        ::mediapipe::Status FillInputTensor(std::vector<std::vector<float>> localFrames);
        void SetOutput(const std::string *str, ::mediapipe::CalculatorContext *cc);
        void DoAfterInference();
        bool DoInference();
        std::vector<std::vector<float>> framesWindow = {};
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        std::tuple<std::string, float> outputWordProb = std::make_tuple("Waiting...", 1.0);
        const std::string LABELS[39] = {"ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي", "ة", "أ", "ؤ", "ئ", "ئـ", "ء", "إ", "آ", "ى", "لا", "ال"};

        int framesSinceLastPrediction = 0;
        int emptyFrames = 0;
        // Options
        bool verboseLog = false;
        int framesWindowSize = 0;
        int thresholdFramesCount = 0;
        int minFramesForInference = 0;
        float probabilitityThreshold = 0.5;
        bool fluentPrediction = false;
        std::string tfLiteModelPath;
        std::unique_ptr<std::future<bool>> inferenceFuture;
    };

    ::mediapipe::Status SignLangPredictionCalculator::GetContract(CalculatorContract *cc)
    {
        RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input has the label " << kLandmarksTag;          
        cc->Inputs().Tag(kLandmarksTag).Set<std::vector<NormalizedLandmarkList>>();

        cc->Outputs().Tag(kTextOutputTag).Set<std::tuple<std::string, float>>();
        return ::mediapipe::OkStatus();
    }
    ::mediapipe::Status SignLangPredictionCalculator::Open(CalculatorContext *cc)
    {
        MP_RETURN_IF_ERROR(LoadOptions(cc)) << "Loading options failed";

        // Load the model
        model = tflite::FlatBufferModel::BuildFromFile(tfLiteModelPath.c_str());

        RET_CHECK(model != nullptr) << "Building model from " << tfLiteModelPath << " failed.";
        
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        interpreter->AllocateTensors();
        if (verboseLog)
        {
            tflite::PrintInterpreterState(interpreter.get());
            LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
            LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
            LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
            LOG(INFO) << "outputs: " << interpreter->outputs().size() << "\n";
            LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";
        }
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status SignLangPredictionCalculator::Process(CalculatorContext *cc)
    {   
        RET_CHECK_OK(UpdateFrames(cc)) << "Updating frames failed.";
        LOG(INFO) << "MEWP3";
        if (inferenceFuture != nullptr && inferenceFuture->wait_for(std::chrono::milliseconds(2)) == std::future_status::ready ) 
        {
            LOG(INFO) << "MEWP2";
            inferenceFuture = nullptr;
            int output_idx = interpreter->outputs()[0];
            float *output = interpreter->typed_tensor<float>(output_idx);
            LOG(INFO) << "MEWP1";
            int highest_pred_idx = -1;
            float highest_pred = 0.0F;
            for (size_t i = 0; i < 39; i++)
            {
                if (verboseLog)
                {
                    LOG(INFO) << LABELS[i] << ": " << *output;
                }
                if (*output > highest_pred)
                {
                    highest_pred = *output;
                    highest_pred_idx = i;
                }
                *output++;
            }
            if (highest_pred > probabilitityThreshold)
            {
                std::string prediction = LABELS[highest_pred_idx];
                outputWordProb = std::make_tuple(prediction, highest_pred);
                cc->Outputs().Tag(kTextOutputTag).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(outputWordProb)
                                                     .At(cc->InputTimestamp()));
            }
            else
            {
                cc->Outputs().Tag(kTextOutputTag).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(std::make_tuple("<unknown>", -1.0)).At(cc->InputTimestamp()));
            }
            return mediapipe::OkStatus();
        }
        if (!ShouldPredict())
        {
            if (fluentPrediction)
            {
                cc->Outputs().Tag(kTextOutputTag).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>().At(cc->InputTimestamp()));
            }
            else
            {
                cc->Outputs()
                    .Tag(kTextOutputTag)
                    .AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(std::make_tuple("Buffer", float(framesWindow.size())))
                                   .At(cc->InputTimestamp()));
            }

            return ::mediapipe::OkStatus();
        }
        // Fill frames up to maximum
        std::vector<std::vector<float>> localFrames = {};
        while (localFrames.size() < framesWindowSize)
        {
            if (framesWindow.size() > localFrames.size())
            {
                localFrames.push_back(framesWindow[localFrames.size()]);
            }
            else
            {
                std::vector<float> frame = {};
                for (size_t i = 0; i < 63; i++)
                {
                    frame.push_back(defaultPoint);
                }
                localFrames.push_back(frame);
            }
        }
        if (inferenceFuture == nullptr)
        {
            RET_CHECK_OK(FillInputTensor(localFrames));
            inferenceFuture = std::make_unique<std::future<bool>>(std::async(std::launch::async, [this]() { return DoInference(); }));
            DoAfterInference();
             cc->Outputs().Tag(kTextOutputTag).AddPacket(mediapipe::MakePacket<std::tuple<std::string, float>>(std::make_tuple("Inference", -1.0)).At(cc->InputTimestamp()));
        }
        
        return ::mediapipe::OkStatus();
    }

    bool SignLangPredictionCalculator::DoInference()
    {
        auto start = std::chrono::high_resolution_clock::now();
        interpreter->Invoke();
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        LOG(INFO) << "Inference time: " << elapsed.count();
        return true;
    }

    ::mediapipe::Status SignLangPredictionCalculator::LoadOptions(
        CalculatorContext *cc)
    {
        const auto &options = cc->Options<SignLangPredictionCalculatorOptions>();
        verboseLog = options.verbose();
        framesWindowSize = options.frameswindowsize();
        thresholdFramesCount = options.thresholdframescount();
        minFramesForInference = options.minframesforinference();
        probabilitityThreshold = options.probabilitythreshold();
        tfLiteModelPath = options.tflitemodelpath();
        fluentPrediction = options.fluentprediction();
        return ::mediapipe::OkStatus();
    }

    void SignLangPredictionCalculator::DoAfterInference()
    {
        framesSinceLastPrediction = 0;
        emptyFrames = 0;
        if (!fluentPrediction)
        {
            framesWindow.clear();
        }
    }

    void SignLangPredictionCalculator::SetOutput(const std::string *str, ::mediapipe::CalculatorContext *cc)
    {
        cc->Outputs()
            .Tag(kTextOutputTag)
            .AddPacket(mediapipe::MakePacket<std::string>(*str)
                           .At(cc->InputTimestamp()));
    }

    ::mediapipe::Status SignLangPredictionCalculator::FillInputTensor(std::vector<std::vector<float>> localFrames)
    {
        int input = interpreter->inputs()[0];
        TfLiteIntArray *dims = interpreter->tensor(input)->dims;
        if (verboseLog)
        {
            LOG(INFO) << "Shape: {" << dims->data[0] << ", " << dims->data[1] << "}";
        }
        float *input_data_ptr = interpreter->typed_input_tensor<float>(0);
        RET_CHECK(input_data_ptr != nullptr);
        for (size_t i = 0; i < localFrames.size(); i++)
        {
            std::vector<float> frame = localFrames[i];
            for (size_t j = 0; j < frame.size(); j++)
            {
                *(input_data_ptr) = frame[j];
                input_data_ptr++;
            }
        }
        return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status SignLangPredictionCalculator::UpdateFrames(CalculatorContext *cc)
    {
        std::vector<float> coordinates = {};

        AddHandDetectionsTo(coordinates, cc);

        if (coordinates.size() < 42)
        { // No hands detected
            if (framesWindow.size() > minFramesForInference)
            {
                emptyFrames++;
            }
            return ::mediapipe::OkStatus();
        }
        int maxSize = 63;
        while (coordinates.size() < maxSize)
        {
            coordinates.push_back(defaultPoint);
        }
        if (coordinates.size() > maxSize)
        {
            LOG(ERROR) << "Coordinates size not equal " << maxSize << ". Actual size: " << coordinates.size();
            return ::mediapipe::OkStatus();
        }

        while (framesWindow.size() >= framesWindowSize)
        {
            framesWindow.erase(framesWindow.begin());
        }

        // Put actual frame into array.
        framesWindow.push_back(coordinates);
        framesSinceLastPrediction++;
        return ::mediapipe::OkStatus();
    }

    bool SignLangPredictionCalculator::ShouldPredict()
    {
        // Minimum frames required for inference
        if (framesSinceLastPrediction < minFramesForInference)
        {
            return false;
        }
        if (fluentPrediction)
        {
            return true;
        }
        // Long enough without hands to predict.
        if (emptyFrames >= thresholdFramesCount)
        {
            return true;
        }
        return false;
    }


    void SignLangPredictionCalculator::AddHandDetectionsTo(std::vector<float> &coordinates, CalculatorContext *cc)
    {
        const std::vector<NormalizedLandmarkList> multiHandLandmarks =
            cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmarkList>>();

        NormalizedLandmarkList landmarks = multiHandLandmarks.at(0);
        for (int i = 0; i < landmarks.landmark_size(); ++i)
        {
            const NormalizedLandmark &landmark = landmarks.landmark(i);
            if (landmark.x() == 0 && landmark.y() == 0)
            {
                continue;
            }
            coordinates.push_back(landmark.x());
            coordinates.push_back(landmark.y());
            coordinates.push_back(landmark.z());
        }
    }


    REGISTER_CALCULATOR(SignLangPredictionCalculator);

} // namespace signlang
