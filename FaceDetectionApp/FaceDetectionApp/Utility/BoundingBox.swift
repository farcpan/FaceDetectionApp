//
//  BoundingBox.swift
//  FaceDetectionApp
//

import Foundation
import UIKit

public class BoundingBox {
    let shapeLayer: CAShapeLayer
    
    init() {
        shapeLayer = CAShapeLayer()
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.lineWidth = 4
        shapeLayer.isHidden = true
    }
    
    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(shapeLayer)
    }
    
    func show(frame: CGRect, color: UIColor) {
        CATransaction.setDisableActions(true)
        
        let path = UIBezierPath(rect: frame)
        shapeLayer.path = path.cgPath
        shapeLayer.strokeColor = color.cgColor
        shapeLayer.isHidden = false
    }
    
    func hide() {
        shapeLayer.isHidden = true
    }
}
