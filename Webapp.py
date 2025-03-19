import gradio as gr
from ultralytics import YOLO
import cv2

# Load model
model = YOLO(r"D:\BCCD\runs\detect\train3\weights\best.pt")

# Function to perform inference
def detect_objects(image):
    results = model(image)
    results[0].save(filename="output.jpg")  # Save result with bounding boxes
    return "output.jpg"

# Gradio interface
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="filepath"),
    title="BCCD Object Detection",
    description="Upload an image to detect RBC, WBC, and Platelets."
)

# Launch app
interface.launch(share=True)
