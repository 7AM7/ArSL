# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")



proto_library(
    name = "sign_lang_prediction_calculator_proto",
    srcs = ["sign_lang_prediction_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["@mediapipe//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "sign_lang_prediction_calculator_cc_proto",
    srcs = ["sign_lang_prediction_calculator.proto"],
    cc_deps = ["@mediapipe//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":sign_lang_prediction_calculator_proto"],
)
cc_library(
    name = "sign_lang_prediction_calculator",
    srcs = ["sign_lang_prediction_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":sign_lang_prediction_calculator_cc_proto",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/formats:landmark_cc_proto",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate"
    ],
    alwayslink = 1,
)

cc_library(
    name = "zscore_per_video_calculator",
    srcs = ["zscore_per_video_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
    ],
    alwayslink = 1,
)