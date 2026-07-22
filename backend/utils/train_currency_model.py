from ultralytics import YOLO

if __name__ == "__main__":
    # Load pretrained YOLOv8 small — best balance of speed vs accuracy
    model = YOLO("yolov8s-cls.pt")   # cls = classification task (genuine vs fake)
    path = r"D:\MNNIT\SAE\V2 ET HACKATHON ANTI\Currency_Dataset\Images"

    model.train(
        data=f"{path}",
        epochs=20,           # increase to 50 if you have time
        imgsz=224,           # image size
        batch=8,
        name="currency_model",
        project="data/runs",
        workers=0
    )

    print("Training done.")