//
//  PostProcess.swift
//  FaceDetectionApp
//

import Foundation
import CoreML

class PostProcess {
    /**
     Calculating the area of rectangle specified by (left, top) and (right, bottom).
     */
    private func area_of(left: Float, top: Float, right: Float, bottom: Float) -> Float {
        var tmpWidth = right - left
        var tmpHeight = bottom - top

        if tmpWidth < 0 { tmpWidth = 0 }
        if tmpHeight < 0 { tmpHeight = 0 }
            
        return tmpWidth * tmpHeight
    }
    
    /**
     Calculating IOU of two rectangles.
        - boxes0 : the rectangles to be compared with size Nx4 (N: the number of rectangles, 4: the number of coordinates of each rectangle)
        - boxes1 : single rectangle with size 1x4
     */
    private func iou_of(boxes0: [Float], boxes1: [Float]) -> [Float] {
        // boxes0: N*4, boxes1: 1*4
        let boxSize = Int(boxes0.count / 4)
        
        /* box1 */
        let box1Left    = boxes1[0]
        let box1Top     = boxes1[1]
        let box1Right   = boxes1[2]
        let box1Bottom  = boxes1[3]
        
        var iou: [Float] = []
        
        for i in 0..<boxSize {
            /* box0 */
            let box0Left    = boxes0[i * 4 + 0]
            let box0Top     = boxes0[i * 4 + 1]
            let box0Right   = boxes0[i * 4 + 2]
            let box0Bottom  = boxes0[i * 4 + 3]
            
            /* Overlap */
            var tmpLeft     = box0Left
            var tmpTop      = box0Top
            var tmpRight    = box0Right
            var tmpBottom   = box0Bottom
            if box1Left     > box0Left   { tmpLeft   = box1Left }
            if box1Top      > box0Top    { tmpTop    = box1Top }
            if box1Right    < box0Right  { tmpRight  = box1Right }
            if box1Bottom   < box0Bottom { tmpBottom = box1Bottom }
            
            let overlapArea = area_of(left: tmpLeft, top: tmpTop, right: tmpRight, bottom: tmpBottom);
            let box0Area = area_of(left: box0Left, top: box0Top, right: box0Right, bottom: box0Bottom);
            let box1Area = area_of(left: box1Left, top: box1Top, right: box1Right, bottom: box1Bottom);

            let tmpIou: Float = overlapArea / (box0Area + box1Area - overlapArea + 0.00001)
            iou.append(tmpIou)
        }
        
        return iou
    }
    
    /**
     Sort argument (index) list of float list.
     ex.) input: [0.1, 0.5, 0.3, 0.2, 0.4] ---> output: [0, 3, 2, 4, 1]
     */
    private func argsort(a : [Float]) -> [Int] {
        return a.enumerated().sorted(by: { $0.element < $1.element }).map({ $0.offset })
    }
    
    /**
     Executing NMS to get Bounding Boxes.
     */
    private func hard_nms(boxes: [Float], scores: [Float], iou_threshold: Float, candidate_size: Int) -> [Float] {
        var picked: [Int] = []
        var indexes = argsort(a: scores)
        if indexes.count > candidate_size {
            indexes = Array(argsort(a: scores)[0..<candidate_size])
        }
                
        while indexes.count > 0 {
            // current index (the last element of array)
            let current = indexes[indexes.count - 1]
            picked.append(current)
            
            if indexes.count == 1 { break }
            
            // without the last element of array
            indexes = Array(indexes[0..<(indexes.count-1)])
            
            var rest_boxes: [Float] = []
            var current_box: [Float] = []
            for index in indexes {
                for j in 0..<4 {
                    rest_boxes.append(boxes[index * 4 + j])
                }
            }
            for j in 0..<4 {
                current_box.append(boxes[current * 4 + j])
            }
                        
            let iou = iou_of(boxes0: rest_boxes, boxes1: current_box)

            var tmp_indexes_list: [Int] = []
            for i in 0..<indexes.count {
                if iou[i] <= iou_threshold {
                    tmp_indexes_list.append(indexes[i])
                }
            }
            indexes = tmp_indexes_list
        }

        var result_box: [Float] = []
        for p in picked {
            for j in 0..<4 {
                result_box.append(boxes[p * 4 + j])
            }
        }
        
        return result_box
    }
    
    func predict(confidences: MLMultiArray, boxes: MLMultiArray, prob_threshold: Float) -> [Float] {
        let class_index = 1
        let probs = confidences.getColumn(column: class_index)
        
        print(probs)
        
        var mask: [Bool] = []
        var subset_probs: [Float] = []
        
        for prob in probs {
            mask.append(prob > prob_threshold)
            if prob > prob_threshold {
                subset_probs.append(prob)
            }
        }
        
        if mask.count <= 0 {
            return [Float]()
        }
        
        var subset_boxes: [Float] = []
        for i in 0..<boxes.shape[1].intValue {
            if mask[i] != true {
                continue
            }
            
            for j in 0..<boxes.shape[2].intValue {
                let value = boxes.getElement(id1: i, id2: j)
                subset_boxes.append(value)
            }
        }
        
        return self.hard_nms(boxes: subset_boxes, scores: subset_probs, iou_threshold: 0.1, candidate_size: 30)
    }
}

/**
 Extension methods for face detection model outputs
 */
extension MLMultiArray {
    /**
     Get the element from MLMultiArray with shape [1, N, m].
        id1: [0, N-1]
        id2: [0, m-1]
     */
    func getElement(id1: Int, id2: Int) -> Float {
        let unsafePointer = UnsafeMutablePointer<Float>(OpaquePointer(self.dataPointer))
        return unsafePointer[id1 * self.shape[2].intValue + id2]
    }
    
    /**
     Get the column from MLMultiArray with shape [1, N, m].
     This method results the array with shape [N].
     */
    func getColumn(column: Int) -> [Float] {
        var tmp: [Float] = []
        for i in 0..<self.shape[1].intValue {
            tmp.append(self.getElement(id1: i, id2: column))
        }
        return tmp
    }
}
