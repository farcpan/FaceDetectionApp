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
    
    var resizedPixelBuffer: CVPixelBuffer?
    
    // MARK:  - AI Model
    let imageWidth = 320
    let imageHeight = 240
    let maxDetection = 200
    let model = face_detector()
    let postProcess = PostProcess()
    
    // MARK: - LifeCycle
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

    // MARK: - Camera Settings
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
    
    // MARK: - Prediction
    /**
     Prediction
     */
    func predict(image: CVPixelBuffer) {
        let startTime = CACurrentMediaTime()
        
        // Resize（w: 320, h: 240）
        guard let resizedImage = resizePixelBuffer(image, width: self.imageWidth, height: self.imageHeight) else { return }
        let result_box = self.predictAndPostProcess(pixelBuffer: resizedImage)
        
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
            
            // Elapsed time measurement
            self.predictionTImeLabel.text = String(format: "Elapsed time: %.5f seconds", CACurrentMediaTime() - startTime)

            // Release semaphore
            self.semaphore.signal()
        }
    }
    
    func predictAndPostProcess(pixelBuffer: CVPixelBuffer) -> [Float] {
        guard let output = try? self.model.prediction(input: pixelBuffer) else {
            return [Float]()
        }

        // postprocessing the prediction result to get boundingboxes
        return self.postProcess.predict(confidences: output.scores, boxes: output.boxes, prob_threshold: 0.8)
    }
}

// MARK: - Camera event delegate
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame frame: CVPixelBuffer?, timestamp: CMTime) {
        // exclusion for video frame extraction
        semaphore.wait()
        
        // prediction
        if let uiImage = frame {
            DispatchQueue.global().async {
                self.predict(image: uiImage)
            }
        }
    }
}
