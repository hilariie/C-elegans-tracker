if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(data="config.yaml", epochs=300, patience=50, device=0, name='c-elegan_tracker', single_cls=False, batch=16)
