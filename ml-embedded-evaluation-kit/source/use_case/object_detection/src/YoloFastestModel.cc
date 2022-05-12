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
#include "YoloFastestModel.hpp"

#include "log_macros.h"

const tflite::MicroOpResolver& arm::app::YoloFastestModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::YoloFastestModel::EnlistOperations()
{
    this->m_opResolver.AddDepthwiseConv2D();
    this->m_opResolver.AddConv2D();
    this->m_opResolver.AddAdd();
    this->m_opResolver.AddResizeNearestNeighbor();
    /*These are needed for UT to work, not needed on FVP */
    this->m_opResolver.AddPad();
    this->m_opResolver.AddMaxPool2D();
    this->m_opResolver.AddConcatenation();
    //for CPU only version
    this->m_opResolver.AddReshape();
    this->m_opResolver.AddSoftmax();


#if defined(ARM_NPU)
    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("Added %s support to op resolver\n",
            tflite::GetString_ETHOSU());
    } else {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
#endif /* ARM_NPU */
    return true;
}

extern uint8_t* GetModelPointer();
const uint8_t* arm::app::YoloFastestModel::ModelPointer()
{
    return GetModelPointer();
}

extern size_t GetModelLen();
size_t arm::app::YoloFastestModel::ModelSize()
{
    return GetModelLen();
}
