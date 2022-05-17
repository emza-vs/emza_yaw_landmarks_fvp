## emza_yaw_landmarks_fvp

## To run the example, clone the ML repo from ARM:

mkdir ~/demo
cd ~/demo

git clone -b 22.02 https://review.mlplatform.org/ml/ethos-u/ml-embedded-evaluation-kit
cd ml-embedded-evaluation-kit
git checkout -b test_branch ed35a6fea4a1604db81c56fc71f7756822fcf212

## clone this repo:
cd ~/demo
git clone https://github.com/emza-vs/emza_yaw_landmarks_fvp.git

## Merge the ml-embedded-evaluation-kit folder from  emza_yaw_landmarks_fvp into the ml-embedded-evaluation-kit folder from ARM,overwrite the files.

## Now go to the modified ml-embedded-evaluation-kit folder and build the example

cd ~/demo/ml-embedded-evaluation-kit
./download_dependencies.py
mkdir build
cd build

cmake .. -DUSE_CASE_BUILD=object_detection -Dobject_detection_IMAGE_SIZE=160 -Dobject_detection_MODEL_TFLITE_PATH=resources_downloaded/object_detection/ssd_slim_120x160x1_yaw_landmarks_v3_int8_vela_H256.tflite -DTARGET_PLATFORM=mps3 -DTARGET_SUBSYSTEM=sse-300  -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake

make

## run the FVP
~/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 -C ethosu.num_macs=256 -a ./bin/ethos-u-object_detection.axf


## NOTE: for detailed step-by step instruction to set-up the GCC and Python toolchanin please look at the readme file here: 

https://github.com/emza-vs/face_detection_example_arm_u55/blob/master/README.md
 


