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