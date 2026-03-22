import cv2
from detector import FruitNinjaDetector

def main():
    detector = FruitNinjaDetector("runs/detect/train/weights/best.pt")
    
    window_name = "Cutting or not?"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Confidence (%)", window_name, 80, 100, lambda x: None)
    cv2.createTrackbar("Overlap (IOU)", window_name, 85, 100, lambda x: None)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success: break

        conf_val = cv2.getTrackbarPos("Confidence (%)", window_name) / 100
        iou_val = cv2.getTrackbarPos("Overlap (IOU)", window_name) / 100

        result_data, current_score = detector.process_frame(frame, conf_val, iou_val)
        
        final_frame = detector.draw_ui(result_data.plot(), current_score, conf_val, iou_val)

        cv2.imshow(window_name, final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()