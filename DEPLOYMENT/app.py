import streamlit as st
import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
import os

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def process_video(input_video_path, output_video_path):
    model = YOLO('yolov8x-seg.pt')
    cap = cv2.VideoCapture(input_video_path)
    START = sv.Point(1250, -2)
    END = sv.Point(1250, 1070)
    track_history = defaultdict(list)
    crossed_objects = {}

    ensure_dir(os.path.dirname(output_video_path))

    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Use H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Alternatively, use 'X264' if 'avc1' doesn't work

        out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height), True)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(frame, conf=0.3, classes=[19], persist=True, save=True, tracker="bytetrack.yaml")
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_frame = results[0].plot() if hasattr(results[0], 'plot') else frame

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append(x)

                if len(track) > 1:  # Checking if we have at least two points to compare
                    if track[-2] <= START.x < track[-1] or track[-2] >= START.x > track[-1]:
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True
                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)
            count_text = f"Objects crossed: {len(crossed_objects)}"
            cv2.putText(annotated_frame, count_text, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

            out.write(annotated_frame)

        cap.release()
        out.release()

def main():
    st.title("Video Processing for Object Tracking")
    video_file = st.file_uploader("Upload a video", type=['mp4', 'avi'])

    if video_file is not None:
        output_folder = "output_videos"
        ensure_dir(output_folder)
        file_path = os.path.join(output_folder, "uploaded_video.mp4")
        file_name = os.path.join(output_folder, "processed_video.mp4")

        # Save the uploaded file first
        with open(file_path, "wb") as f:
            f.write(video_file.getbuffer())

        if st.button("Process Video"):
            process_video(file_path, file_name)
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
