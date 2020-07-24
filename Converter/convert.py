import argparse
import onnx
import os

from onnx_coreml import convert


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_model_path', help='The path of onnx model to be converted.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    model_path = args.onnx_model_path

    if not os.path.exists(args.onnx_model_path):
        print("Model file <{}> does not exists.".format(model_path))
    else:
        model_onnx = onnx.load_model(model_path)

        # conversion
        mlmodel = convert(model_onnx, 
            model='regression', 
            image_input_names=['image'], 
            minimum_ios_deplyment_target='13')

        mlmodel.save('face_detector.mlmodel')