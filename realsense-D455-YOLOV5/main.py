import argparse
from detect import Detect  # 引入封装好的类
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    # Create Detect instance
    detector = Detect(weights=opt.weights, device=opt.device, iou=opt.iou_thres, source='0', view_img=opt.view_img, save_txt=opt.save_txt, img_size=opt.img_size)
    detector.initialize(save_dir=opt.save_dir)

    try:
        while True:
            color_image, depth_frame, pred, img = detector.detect(conf={
                'img_size': opt.img_size,
                'augment': opt.augment,
                'conf_thres': opt.conf_thres,
                'classes': opt.classes,
                'agnostic_nms': opt.agnostic_nms
            })
            if color_image is not None and pred is not None:
                detector.plot_and_save(frame=color_image, depth_frame=depth_frame, pred=pred, img=img, save_dir=opt.save_dir)
                print("www")
                
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
                

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
