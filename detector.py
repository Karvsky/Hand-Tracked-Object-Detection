import cv2

class FruitNinjaDetector:
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.pom1 = 0
        self.counter = 0

    def process_frame(self, frame, conf, iou):
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            agnostic_nms=True,
            device=0,
            verbose=False
        )

        classes = results[0].boxes.cls.cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        
        cutting_confs = [conf for cls, conf in zip(classes, confidences) if cls == 1]
        max_cutting_conf = max(cutting_confs) if cutting_confs else 0

        if 0 in classes: 
            self.pom1 = 1
            
        if max_cutting_conf > 0.60 and self.pom1 == 1:
            self.counter += 1
            self.pom1 = 0

        return results[0], self.counter

    def draw_ui(self, frame, counter, conf_val, iou_val):
        annotated_frame = frame.copy()
        h, w, _ = annotated_frame.shape

        cv2.putText(annotated_frame, f"CONF: {conf_val} IOU: {iou_val}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        text = f"Cutting: {counter}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = w - text_w - 20
        text_y = text_h + 20

        cv2.putText(annotated_frame, text, (text_x, text_y), font, 
                    font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        
        return annotated_frame