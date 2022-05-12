/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DETECTOR_POST_PROCESSING_HPP
#define DETECTOR_POST_PROCESSING_HPP

#include "UseCaseCommonUtils.hpp"
#include "ImageUtils.hpp"
#include "DetectionResult.hpp"
#include "YoloFastestModel.hpp"

#include <forward_list>


#define SSD_NUM_PRIORS 1118
#define SSD_NUM_FEATURE_MAPS 4
#define SSD_NUM_MAX_TARGETS 20
#define WITH_YAW 0
#define WITH_YAW_AND_LANDMARKS 1

namespace arm {
namespace app {
namespace object_detection {

    struct Branch {
        int resolution;
        int numBox;
        const float* anchor;
        int8_t* modelOutput;
        float scale;
        int zeroPoint;
        size_t size;
    };

    struct Network {
        int inputWidth;
        int inputHeight;
        int numClasses;
        std::vector<Branch> branches;
        int topN;
    };

    /**
     * @brief   Helper class to manage tensor post-processing for "object_detection"
     *          output.
     */
    class DetectorPostprocessing {
    public:
        /**
         * @brief       Constructor.
         * @param[in]   threshold     Post-processing threshold.
         * @param[in]   nms           Non-maximum Suppression threshold.
         * @param[in]   numClasses    Number of classes.
         * @param[in]   topN          Top N for each class.
         **/
        DetectorPostprocessing(float threshold = 0.5f,
                                float nms = 0.45f,
                                int numClasses = 1,
                                int topN = 0);

        /**
         * @brief       Post processing part of Yolo object detection CNN.
         * @param[in]   imgIn        Pointer to the input image,detection bounding boxes drown on it.
         * @param[in]   imgRows      Number of rows in the input image.
         * @param[in]   imgCols      Number of columns in the input image.
         * @param[in]   modelOutput  Output tensors after CNN invoked.
         * @param[out]  resultsOut   Vector of detected results.
         **/
        void RunPostProcessing(uint8_t* imgIn,
                               uint32_t imgRows,
                               uint32_t imgCols,
                               TfLiteTensor* modelOutput0,
                               TfLiteTensor* modelOutput1,
                               std::vector<DetectionResult>& resultsOut);

    private:
        float m_threshold;  /* Post-processing threshold */
        float m_nms;        /* NMS threshold */
        int   m_numClasses; /* Number of classes */
        int   m_topN;       /* TopN */

        /**
         * @brief       Insert the given Detection in the list.
         * @param[in]   detections   List of detections.
         * @param[in]   det          Detection to be inserted.
         **/
        void InsertTopNDetections(std::forward_list<image::Detection>& detections, image::Detection& det);

        /**
         * @brief        Given a Network calculate the detection boxes.
         * @param[in]    net           Network.
         * @param[in]    imageWidth    Original image width.
         * @param[in]    imageHeight   Original image height.
         * @param[in]    threshold     Detections threshold.
         * @param[out]   detections    Detection boxes.
         **/
        void GetNetworkBoxes(Network& net,
                             int imageWidth,
                             int imageHeight,
                             float threshold,
                             std::forward_list<image::Detection>& detections);

        /**
         * @brief       Draw on the given image a bounding box starting at (boxX, boxY).
         * @param[in/out]   imgIn    Image.
         * @param[in]       imWidth    Image width.
         * @param[in]       imHeight   Image height.
         * @param[in]       boxX       Axis X starting point.
         * @param[in]       boxY       Axis Y starting point.
         * @param[in]       boxWidth   Box width.
         * @param[in]       boxHeight  Box height.
         **/
        void DrawBoxOnImage(uint8_t* imgIn,
                            int imWidth,
                            int imHeight,
                            int boxX,
                            int boxY,
                            int boxWidth,
                            int boxHeight);
    };

} /* namespace object_detection */

namespace ssd {

typedef struct EmzaColor {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} EmzaColor;

typedef struct FaceInfo {
    short x1;
    short y1;
    short x2;
    short y2;
    #if WITH_YAW
    short yaw;
    #endif // WITH_YAW
    #if WITH_YAW_AND_LANDMARKS
    short yaw;
    short landmarks[10]; // (x,y ... ,x,y)
    #endif // WITH_YAW_AND_LANDMARKS
    float score;
} FaceInfo;


short generateBBox(TfLiteTensor* model_output[], FaceInfo out_boxes[]);

short nms(FaceInfo *input_boxes, short num_input_boxes, FaceInfo output_boxes[]);

void drawLandmarksAndYaw(uint8_t* srcPtr,int srcHeight, int srcWidth,
                         FaceInfo *face_list, int n_faces,int n_channels);

} /* namepsace ssd */

} /* namespace app */
} /* namespace arm */

#endif /* DETECTOR_POST_PROCESSING_HPP */
