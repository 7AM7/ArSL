#include "mediapipe/framework/calculator_framework.h"

#include <vector>


namespace mediapipe{

    namespace{
        constexpr char LandmarksHistory[] = "DOUBLE_LANDMARKS_HISTORY";
    }


    class ZscorePerVideoCalculator : public CalculatorBase {
        public:
        ZscorePerVideoCalculator(){};
        ~ZscorePerVideoCalculator(){};

        static ::mediapipe::Status GetContract(CalculatorContract* cc){
            cc->Inputs().Tag(LandmarksHistory).Set<std::vector<std::vector<double>>>();
            cc->Outputs().Tag(LandmarksHistory).Set<std::vector<std::vector<double>>>();
            return ::mediapipe::OkStatus();
        }
        ::mediapipe::Status Open(CalculatorContext* cc){

            return ::mediapipe::OkStatus();
        }
        ::mediapipe::Status Process(CalculatorContext* cc){
            std::vector<std::vector<double>> frames = cc->Inputs().Tag(LandmarksHistory).Get<std::vector<std::vector<double>>>();
            float x_mean = 0;
            float x_sdev = 0;
            float y_mean = 0;
            float y_sdev = 0;
            for(int i = 0; i < frames.size(); i++){
                auto frame = frames.at(i);
                for(int j = 0; j < 21; j++){
                    x_mean += frame.at(j*2);
                    y_mean += frame.at(j*2+1);
                }
            }
            int coordinate_count = 21 * frames.size();
            if(coordinate_count == 0){
                coordinate_count = 1;
            }
            x_mean /= coordinate_count;
            y_mean /= coordinate_count;
            for(int i = 0; i < frames.size(); i++){
                auto frame = frames.at(i);
                for(int j = 0; j < 21; j++){
                    x_sdev += powf(frame.at(j*2) - x_mean, 2.0);
                    y_sdev += powf(frame.at(j*2+1) - x_mean, 2.0);
                }
            }
            x_sdev /= coordinate_count;
            y_sdev /= coordinate_count;

            x_sdev = sqrtf(x_sdev);
            y_sdev = sqrtf(y_sdev);
            if(x_sdev == 0){
                x_sdev = 1;
            }
            if(y_sdev == 0){
                y_sdev = 1;
            }

            std::vector<std::vector<double>> output_frames;
            for(int i = 0; i < frames.size(); i++){
                std::vector<double> output_frame;
                for(int j = 0; j < 21; j++){
                    output_frame.push_back((frames.at(i).at(j*2) - x_mean) / x_sdev);
                    output_frame.push_back((frames.at(i).at(j*2+1) - y_mean) / y_sdev);
                }
                output_frames.push_back(output_frame);
            }
            if(output_frames.size() != frames.size()){
                LOG(ERROR) << "Zscore Per Video does not work";
            }


            std::unique_ptr<std::vector<std::vector<double>>> output_stream_collection = std::make_unique<std::vector<std::vector<double>>>(output_frames); 
            cc -> Outputs().Tag(LandmarksHistory).Add(output_stream_collection.release(), cc->InputTimestamp());
            return ::mediapipe::OkStatus();
        }
        ::mediapipe::Status Close(CalculatorContext* cc){
            return ::mediapipe::OkStatus();
        }

        private:

    };
    
    REGISTER_CALCULATOR(ZscorePerVideoCalculator);
}