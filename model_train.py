from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11n.pt")

    result = model.train(data="./data.yaml", epochs=40, imgsz=640, device=0, batch=16)

    return result

def trained_model():

    model = YOLO("./runs/detect/train5/weights/best.pt")

    return model

if __name__ == "__main__":

    training = train_model()
    model = trained_model()