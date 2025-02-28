import multiprocessing
import os
import cv2
import time
from ultralytics import YOLO
start_time = time.time()
num_processes = multiprocessing.cpu_count()
print(num_processes)
def detect_image(video_path, model_path, output_path):

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read the first frame from {video_path}")
        return

    H, W, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    # Load a model
    model = YOLO(model_path)  # load a custom model
    threshold = 0.5

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()


def process_video(video_path, model_path):
    output_path = f'output_{os.path.basename(video_path)}'
    detect_image(video_path, model_path, output_path)

if __name__ == "__main__":
    # Define your video and model paths
    path = 'H:/Museum/Museum Project/60_output_process_1/'
    video_paths = [path+'part_1.mp4']  # Add more video paths if needed
    model_path = 'last.pt'

    processes = []
    for video_path in video_paths:
        process = multiprocessing.Process(target=process_video, args=(video_path, model_path))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All videos processed.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Processing completed for. Time: {execution_time:.2f} seconds")
    with open(f'execution_time.txt', 'w') as file:
        file.write(f'Execution Time: {execution_time:.2f} seconds')
