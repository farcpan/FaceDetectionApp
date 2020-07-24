//
//  ViewController.swift
//  FaceDetectionApp
//

import AVFoundation
import CoreImage
import CoreML
import UIKit
import VideoToolbox

class ViewController: UIViewController {
    // MARK: - UI
    @IBOutlet weak var cameraView: UIImageView!
    @IBOutlet weak var predictionTImeLabel: UILabel!
    var boundingBoxes: [BoundingBox] = []
    
    // MARK: - Camera
    var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 2)
    
    // MARK:  - AI Model
    let imageWidth = 320
    let imageHeight = 240
    let maxDetection = 200
    let model = face_detector()
    let postProcess = PostProcess()
    
    // MARK: - ライフサイクル
    override func viewDidLoad() {
        super.viewDidLoad()
        // 顔検出領域矩形初期化
        self.setUpBoundingBoxes()
        // カメラ設定
        self.setupCamera()
    }
    
    // MARK: - Bounding Box
    func setUpBoundingBoxes() {
        for _ in 0..<self.maxDetection {
            self.boundingBoxes.append(BoundingBox())
        }
    }

    // MARK: - カメラ設定
    func setupCamera() {
        self.videoCapture = VideoCapture()
        self.videoCapture.delegate = self
        self.videoCapture.fps = 30
        self.videoCapture.setup(sessionPreset: AVCaptureSession.Preset.vga640x480) { success in
            if success {
                // Add the video preview into the UI.
                if let previewLayer = self.videoCapture.previewLayer {
                    self.cameraView.layer.addSublayer(previewLayer)
                    self.videoCapture.previewLayer?.frame = self.cameraView.bounds
                }
                
                // BoundingBox設定
                for box in self.boundingBoxes {
                    box.addToLayer(self.cameraView.layer)
                }
                
                // 動画フレーム取得開始
                self.videoCapture.start()
            }
        }
    }
    
    // MARK: - 推論
    /**
     推論実行
     */
    func predict(image: UIImage) {
        let startTime = CACurrentMediaTime()
        
        // 画像リサイズ（w: 320, h: 240）
        let resizedImage = image.resize(to: CGSize(width: self.imageWidth, height: self.imageHeight))
        // 推論実行
        let result_box = self.predictAndPostProcess(image: resizedImage)
        
        // on MainThread
        DispatchQueue.main.async {
            let width = Int(self.cameraView.bounds.width)
            let height = Int(self.cameraView.bounds.height)
            
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
                
                let rect: CGRect = CGRect(x: Int(x), y: Int(y), width: Int(w * 1.1), height: Int(h * 1.1))
                self.boundingBoxes[i].show(frame: rect, color: UIColor(red: 0.75, green: 0.1, blue: 0.25, alpha: 0.9))
            }
            
            // 推論時間測定
            self.predictionTImeLabel.text = String(format: "Elapsed time: %.5f seconds", CACurrentMediaTime() - startTime)
            
            // Release semaphore
            self.semaphore.signal()
        }
    }
    
    func predictAndPostProcess(image: UIImage) -> [Float] {
        let pixelBuffer = image.getPixelBuffer()
        let mlarray = try! MLMultiArray(
            shape: [1, NSNumber(value: 3), NSNumber(value: self.imageHeight), NSNumber(value: self.imageWidth)], dataType: MLMultiArrayDataType.float32
        )
        
        for i in 0..<3*imageHeight*imageWidth {
            mlarray[i] = pixelBuffer[i] as NSNumber
        }
        
        guard let output = try? self.model.prediction(input: mlarray) else {
            return [Float]()
        }

        return self.postProcess.predict(confidences: output.scores, boxes: output.boxes, prob_threshold: 0.8)
    }
}

// MARK: - カメライベントデリゲート
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame frame: UIImage?, timestamp: CMTime) {
        // 動画フレーム排他制御
        semaphore.wait()
        
        // 推論実行
        if let uiImage = frame {
            DispatchQueue.global().async {
                self.predict(image: uiImage)
            }
        }
    }
}

// MARK: - UIImage Extension
extension UIImage {
    /**
     UIImageをリサイズする
     */
    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    /**
     PixelBufferをFloat配列に変換する
     */
    func getPixelBuffer() -> [Float] {
        guard let cgImage = self.cgImage else {
            return []
        }
        
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data

        var buf: [Float] = []
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
