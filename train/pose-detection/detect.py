import cv2 
import argparse

from ultralytics.models.yolo.pose import PosePredictor


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov8n-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='ultralytics/assets', help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    return opt


def main(opt):
    args = dict(model=opt.weights, source=opt.source)
    predictor = PosePredictor(overrides=args)
    predictor.predict_cli()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
