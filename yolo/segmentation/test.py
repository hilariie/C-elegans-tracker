if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO('runs/segment/train/weights/best.pt')
    metrics = model.val(data='test.yaml')
    print(metrics.results_dict)
