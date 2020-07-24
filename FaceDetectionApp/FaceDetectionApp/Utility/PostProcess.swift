//
//  PostProcess.swift
//  FaceDetectionApp
//

import Foundation

public class PostProcess {
    private func get_score(id1: Int, id2: Int, array: [Float]) -> Float {
        return array[id1 * 2 + id2]
    }

    private func get_box(id1: Int, id2: Int, array: [Float]) -> Float {
        return array[id1 * 4 + id2]
    }
    
    private func get_score_column(column: Int, array: [Float]) -> [Float] {
        var tmp: [Float] = []
        for i in 0..<4420 {
            let value = get_score(id1: i, id2: column, array: array)
            tmp.append(value)
        }
        return tmp
    }

    private func area_of(left: Float, top: Float, right: Float, bottom: Float) -> Float {
        var tmpWidth = right - left
        var tmpHeight = bottom - top

        if tmpWidth < 0 { tmpWidth = 0 }
        if tmpHeight < 0 { tmpHeight = 0 }
            
        return tmpWidth * tmpHeight
    }
    
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
    
    private func argsort(a : [Float]) -> [Int] {
        return a.enumerated().sorted(by: { $0.element < $1.element }).map({ $0.offset })
    }
    
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
    
    func predict(confidences: [Float], boxes: [Float], prob_threshold: Float) -> [Float] {
        let class_index = 1

        let probs = get_score_column(column: class_index, array: confidences)
        
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
        for i in 0..<4420 {
            if mask[i] != true {
                continue
            }
            
            for j in 0..<4 {
                let value = get_box(id1: i, id2: j, array: boxes)
                subset_boxes.append(value)
            }
        }
        
        return self.hard_nms(boxes: subset_boxes, scores: subset_probs, iou_threshold: 0.1, candidate_size: 100)
    }
}
