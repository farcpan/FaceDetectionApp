# FaceDetectionApp

顔検出iOSアプリケーション。iOS13以降のみ対応。


---

## 使用手順

1. [pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)から[ONNXモデル](https://github.com/cunjian/pytorch_face_landmark/blob/master/models/onnx/version-RFB-320.onnx)を入手する。

1. ONNXモデルをCoreMLモデルに変換する。

    ```
    $ cd Converter
    $ python convert.py <ONNX Model Path>/version-RFB-320.onnx
    $ mv face_detector.mlmodel ../FaceDetectionApp/FaceDetectionApp/Model/
    ```

1. iOSアプリケーションプロジェクトをビルドする。

---