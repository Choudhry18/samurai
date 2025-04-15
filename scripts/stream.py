import cv2
from sam2.build_sam import build_sam2_video_predictor_hf
import numpy as np
import os
import torch
import gc
import argparse
import threading
import time
import queue

def get_gpu_memory_usage():
    """Get the current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.memory_reserved() / 1024 / 1024
    return 0, 0

def log_memory(tag="", log_mem=False):
    if not log_mem:
        return 
    """Log memory usage with an optional tag."""
    allocated, reserved = get_gpu_memory_usage()
    print(f"[{tag}] GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")


def frame_producer(video_source, frame_queue, stop_event):
    """Capture frames from the video source and put them in the queue."""
    cap = cv2.VideoCapture(video_source)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
        time.sleep(1 / 30)  # Simulate 30 FPS capture rate
    cap.release()
    stop_event.set()  # Signal the consumer to stop

def frame_consumer(frame_queue, predictor, inference_state, output_path, stop_event, args):
    """Perform inference on frames from the queue and visualize results."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))  # Adjust width/height as needed

    # Define color for visualization
    color = (0, 255, 0)  # Green for masks and bounding boxes

    # Example bounding box (x1, y1, x2, y2)
    bbox = (250, 50, 420, 320)

    # Add the bounding box to the predictor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        predictor.add_new_points_or_box(inference_state, box=bbox, frame_idx=0, obj_id=0)

    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=0.1)  # Wait for a frame
        except queue.Empty:
            continue

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            frame_idx, object_ids, masks = predictor.propagate_streaming(inference_state, frame)

            # Create a copy for visualization
            output_frame = frame.copy()

            # Visualize the masks and bounding boxes
            for obj_id, mask in zip(object_ids, masks):
                mask_binary = mask[0].cpu().numpy() > 0

                # Create a colored mask overlay
                mask_overlay = np.zeros_like(output_frame, dtype=np.uint8)
                mask_overlay[mask_binary] = color
                output_frame = cv2.addWeighted(output_frame, 1, mask_overlay, 0.5, 0)

                # Calculate bounding box from the mask
                non_zero_indices = np.argwhere(mask_binary)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0)
                    y_max, x_max = non_zero_indices.max(axis=0)
                    cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), color, 2)

            out.write(output_frame)  # Write the processed frame to the output file

    out.release()

def main(args):
    # Determine video source
    video_source = args.video if args.video else args.camera

    # Initialize model
    log_memory("Before model load", args.log_mem)
    predictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-base-plus", device="cuda:0")
    log_memory("After model load", args.log_mem)

    # Initialize streaming state
    cap = cv2.VideoCapture(video_source)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Unable to read the first frame.")
        return

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        inference_state = predictor.init_streaming_state(first_frame)

    # Setup producer-consumer
    frame_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory issues
    stop_event = threading.Event()

    producer_thread = threading.Thread(target=frame_producer, args=(video_source, frame_queue, stop_event))
    consumer_thread = threading.Thread(target=frame_consumer, args=(frame_queue, predictor, inference_state, "output.mp4", stop_event, args))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    # Cleanup
    del predictor, inference_state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    print("Processing complete. Video saved to output.mp4.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2 video predictor on a video file or camera.")
    parser.add_argument("--video", type=str, default=None, help="Path to the video file. If not provided, the camera will be used.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use (default: 0). Ignored if --video is provided.")
    parser.add_argument("--log_mem", type=bool, default=False, help="Log GPU memory usage.")
    args = parser.parse_args()
    main(args)