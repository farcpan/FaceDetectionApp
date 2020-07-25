//
//  VideoCapture.swift
//  FaceDetectionApp
//
//  This code is from:
//      https://github.com/Ma-Dan/YOLOv3-CoreML/blob/master/YOLOv3-CoreML/YOLOv3-CoreML/VideoCapture.swift

import UIKit
import AVFoundation
import CoreVideo

public protocol VideoCaptureDelegate: class {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame: CVPixelBuffer?, timestamp: CMTime)
}

public class VideoCapture: NSObject {
    public var previewLayer: AVCaptureVideoPreviewLayer?
    public weak var delegate: VideoCaptureDelegate?
    public var fps = 5
    
    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    let queue = DispatchQueue(label: "video_stream.camera-queue")
    
    var lastTimestamp = CMTime()
    
    public func setup(sessionPreset: AVCaptureSession.Preset = .medium, completion: @escaping (Bool) -> Void) {
        queue.async {
            let success = self.setupCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }
    
    func setupCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset
        
        guard let captureDevice = AVCaptureDevice.default(for: AVMediaType.video) else { return false }
        
        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else { return false }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect
        previewLayer.connection?.videoOrientation = .portrait
        self.previewLayer = previewLayer
        
        let settings: [String : Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA),
        ]
        
        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        videoOutput.connection(with: AVMediaType.video)?.videoOrientation = .portrait
        
        captureSession.commitConfiguration()
        return true
    }
    
    public func start() {
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
    }
    
    public func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let deltaTime = timestamp - lastTimestamp

        if deltaTime >= CMTimeMake(value: 1, timescale: Int32(self.fps)) {
            lastTimestamp = timestamp
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

            delegate?.videoCapture(self, didCaptureVideoFrame: pixelBuffer, timestamp: timestamp)
        }
    }
}
