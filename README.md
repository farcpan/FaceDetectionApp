# FaceDetectionApp

Face detection application for iOS (>=iOS 13.0).

---

## How to start project

1. Convert `ONNX` model to CoreML model. You can get `ONNX` model file (`version-RFB-320.onnx`) from [pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark). 
    ```
    $ cd Converter
    $ python convert.py ./version-RFB-320.onnx
    $ mv face_detector.mlmodel ../FaceDetectionApp/FaceDetectionApp/Model/
    ```

1. Build iOS Project. 

---

## Model abstract

Face Detection model `version-RFB-320.onnx` has input with shape `[1, 3, 240, 320]` and outputs with shape `[1, 4420, 2]` (scores) and `[1, 4420, 4]` (bounding box candidates). 

The conversion script (`Converter/convert.py`) includes the preprocessing to normalize input image pixel value range `[0, 255]` (for iOS) to `[-1, 1]` (for onnx model) by the `preprocessing_args` parameter for `convert` API in `onnx_coreml`. 

---