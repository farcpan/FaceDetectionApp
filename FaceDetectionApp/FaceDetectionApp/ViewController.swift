//
//  ViewController.swift
//  FaceDetectionApp
//

import AVFoundation
import CoreImage
import CoreML
import UIKit
import Vision

class ViewController: UIViewController {
    // MARK: - UI
    @IBOutlet weak var cameraView: UIImageView!
    
    // MARK: - Camera
    var captureSession = AVCaptureSession()
    var camera: AVCaptureDevice?
    var photoOutput: AVCapturePhotoOutput?
    var cameraPreviewLayer: AVCaptureVideoPreviewLayer?
    
    // MARK:  - AI Model
    let imageWidth = 320
    let imageHeight = 240
    let maxDetection = 200
    
    let model = face_detector()
    let postProcess = PostProcess()
    
    // MARK: - BoundingBox
    var boundingBoxes: [BoundingBox] = []
    
    func setUpBoundingBoxes() {
        for _ in 0..<self.maxDetection {
            self.boundingBoxes.append(BoundingBox())
        }
    }
    
    // MARK: - ライフサイクル
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        // カメラ設定
        self.setupCaptureSession()
        self.setupDevice()
        self.setupInputOutput()
        self.setupPreviewLayer()
        
        // BoundingBox設定
        self.setUpBoundingBoxes()
        for box in self.boundingBoxes {
            box.addToLayer(self.cameraView.layer)
        }
        
        // カメラ処理開始
        self.captureSession.startRunning()
    }
    
    // MARK: - イベント
    /**
     ボタンプッシュイベント
     */
    @IBAction func push(_ sender: Any) {
        let settings = AVCapturePhotoSettings()
        settings.flashMode = .auto
        // settings.isAutoStillImageStabilizationEnabled = true
        
        self.photoOutput?.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }
    
    // MARK: - 推論
    /**
     推論実行
     */
    func predict(image: UIImage, width: Int, height: Int) {
        let pixelBuffer = image.getPixelBuffer()
        let mlarray = try! MLMultiArray(
            shape: [1, NSNumber(value: 3), NSNumber(value: self.imageHeight), NSNumber(value: self.imageWidth)], dataType: MLMultiArrayDataType.float32
        )
        
        for i in 0..<3*imageHeight*imageWidth {
            mlarray[i] = pixelBuffer[i] as NSNumber
        }
        
        if let output = try? self.model.prediction(input: mlarray)  {
            var boxesArray: [Float] = []
            var confidencesArray: [Float] = []
            
            for i in 0..<output.boxes.count {
                boxesArray.append(Float(truncating: output.boxes[i]))
            }
            for i in 0..<output.scores.count {
                confidencesArray.append(Float(truncating: output.scores[i]))
            }
            
            let result_box = self.postProcess.predict(
                width: width, height: height, confidences: confidencesArray, boxes: boxesArray, prob_threshold: 0.8)
            
            // BoundingBox
            let numOfBoundingBox = Int(result_box.count / 4)
            for box in self.boundingBoxes {
                box.hide()  // clear previous results
            }
            for i in 0..<numOfBoundingBox {
                let x = result_box[i * 4 + 0] * Float(width)
                let y = result_box[i * 4 + 1] * Float(height)
                let w = result_box[i * 4 + 2] * Float(width) - x
                let h = result_box[i * 4 + 3] * Float(height) - y
                let rect: CGRect = CGRect(x: Int(x), y: Int(y), width: Int(w), height: Int(h))
                self.boundingBoxes[i].show(frame: rect, color: UIColor(red: 0.9, green: 0.1, blue: 0.1, alpha: 0.9))
            }
        }
    }
    
    // MARK: - カメラ設定
    /**
     カメラ画質設定
     */
    func setupCaptureSession() {
        captureSession.sessionPreset = AVCaptureSession.Preset.medium
    }
    
    /**
     カメラ初期設定
     */
    func setupDevice() {
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [AVCaptureDevice.DeviceType.builtInWideAngleCamera],
            mediaType: AVMediaType.video,
            position: AVCaptureDevice.Position.unspecified
        )
        
        let devices = deviceDiscoverySession.devices
        for device in devices {
            if device.position == AVCaptureDevice.Position.back {
                camera = device
                break
            }
        }
    }
    
    /**
     カメラ入出力設定
     */
    func setupInputOutput() {
        do {
            let captureDeviceInput = try AVCaptureDeviceInput(device: camera!)
            captureSession.addInput(captureDeviceInput)
            photoOutput = AVCapturePhotoOutput()
            photoOutput!.setPreparedPhotoSettingsArray(
                [AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])],
                completionHandler: nil)
            
            captureSession.addOutput(photoOutput!)
        } catch {
            print(error)
        }
    }
    
    /**
     カメラプレビューレイヤー追加
     */
    func setupPreviewLayer() {
        self.cameraPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
        self.cameraPreviewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        self.cameraPreviewLayer?.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
        
        self.cameraPreviewLayer?.frame = self.cameraView.frame
        self.cameraView.layer.insertSublayer(self.cameraPreviewLayer!, at: 0)
    }
}

//MARK: - カメライベントデリゲート
extension ViewController: AVCapturePhotoCaptureDelegate {
    /**
     撮影した画像データが生成されたときに呼び出されるデリゲートメソッド
     */
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        // 撮影した画像を取得
        guard let imageData = photo.fileDataRepresentation() else { return }
        // Data型をUIImageオブジェクトに変換
        guard let uiImage = UIImage(data: imageData) else { return }
        // 画像リサイズ
        let resizedImage = uiImage.resize(to: CGSize(width: self.imageWidth, height: self.imageHeight))
        // 推論実行
        self.predict(image: resizedImage, width: Int(uiImage.size.width), height: Int(uiImage.size.height))
    }
}

//MARK: - UIImage Extension
extension UIImage {
    /*
     UIImageをリサイズする
     */
    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    
    func getPixelBuffer() -> [Float] {
        guard let cgImage = self.cgImage else {
            return []
        }
        
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data

        var buf : [Float] = []
        for c in 0..<3 {
            for i in 0..<height {
                for j in 0..<width {
                    let pixelInfo = bytesPerRow * i + j * bytesPerPixel + c
                    let pixel = (CGFloat(pixelData[pixelInfo]) - 127.0) / 128.0
                    buf.append(Float(pixel))
                }
            }
        }
        return buf
    }
}
