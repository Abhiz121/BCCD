from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data=r"D:\BCCD\Dataset\bccd\data.yaml",
    epochs=50,        
    imgsz=640,        
    batch=8,          
    device="cuda"  
)

print("Training Completed!")