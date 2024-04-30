import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os
from collections import defaultdict

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (int(point[0]), int(point[1])), False) > 0

def process_video(input_video_path, output_video_path, polygons):
    model = YOLO('yolov8x-seg.pt')
    cap = cv2.VideoCapture(input_video_path)
    intruder_ids = set()

    ensure_dir(os.path.dirname(output_video_path))

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height), True)

        poly_arrays = [np.array(polygon, np.int32) for polygon in polygons]
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, conf=0.6, classes=None, persist=True, save=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().numpy()

            annotated_frame = results[0].plot() if hasattr(results[0], 'plot') else frame

            for poly_array in poly_arrays:
                cv2.polylines(annotated_frame, [poly_array], True, (0, 255, 0), 3)

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                centroid = (x + w / 2, y + h / 2)
                
                if cls != 19:
                    for poly_array in poly_arrays:
                        if is_point_in_polygon(centroid, poly_array):
                            if track_id not in intruder_ids:
                                intruder_ids.add(track_id)
                                cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                                cv2.putText(annotated_frame, f"Intruder ID: {track_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            break

            intrusion_text = f"Total Intruders Detected: {len(intruder_ids)}"
            cv2.putText(annotated_frame, intrusion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            out.write(annotated_frame)

        cap.release()
        out.release()

def main():
    st.title("Video Processing for Intrusion Detection")
    video_file = st.file_uploader("Upload a video", type=['mp4', 'avi'])

    if video_file is not None:
        output_folder = "output_videos"
        ensure_dir(output_folder)
        file_path = os.path.join(output_folder, "uploaded_video.mp4")
        file_name = os.path.join(output_folder, "processed_video.mp4")

        with open(file_path, "wb") as f:
            f.write(video_file.getbuffer())

        polygons = [
            
np.array([
[110, 70],[1754, 62],[1754, 1062],[138, 1066],[110, 70]
])

    
        ]

        if st.button("Process Video"):
            process_video(file_path, file_name, polygons)
            st.video(file_name)

            with open(file_name, "rb") as file:
                st.download_button(
                    label="Download processed video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()
