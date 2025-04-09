import cv2
from sam2.build_sam import build_sam2_video_predictor_hf
import numpy as np
import os
import torch
import gc
import argparse

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

def frame_generator(video_source):
    """Generate frames from a video source."""
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()
    return fps

def main(args):
    # Determine video source
    if args.video:
        video_source = args.video
        if not os.path.exists(video_source):
            print(f"Error: Video file not found at {os.path.abspath(video_source)}")
            return
    else:
        video_source = args.camera

    # Initialize model
    log_memory("Before model load", args.log_mem)
    predictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-base-plus", device="cuda:0")
    log_memory("After model load", args.log_mem)
    
    # Setup streaming source
    frame_gen = frame_generator(video_source)
    
    # Get video properties for output file
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) if args.video else 30  # Default to 30 FPS for camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    first_frame = next(frame_gen)
        
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        inference_state = predictor.init_streaming_state(first_frame)

    bbox = (250, 50, 420, 320)  # (x1, y1, x2, y2)

    # Define color for visualization
    color = [(255, 0, 0)]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        predictor.add_new_points_or_box(inference_state, box=bbox, frame_idx=0, obj_id=0)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while True:
            try:
                new_frame = next(frame_gen)
                frame_idx, object_ids, masks = predictor.propagate_streaming(inference_state, new_frame)

                # Create a copy for visualization
                output_frame = new_frame.copy()
                mask_to_vis = {}
                bbox_to_vis = {}

                if frame_idx % 10 == 0:
                    log_memory(f"Before propagate_streaming frame {frame_idx}", args.log_mem)
                # Visualize the masks
                for obj_id, mask in zip(object_ids, masks):  
                    mask_binary = mask[0].cpu().numpy() > 0
                    
                    # Create a colored mask overlay
                    non_zero_indices = np.argwhere(mask_binary)  # Find non-zero pixels (object pixels)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]  # If no object is found, set bbox to zero
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()  # Top-left corner
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()  # Bottom-right corner
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # Convert to (x, y, w, h)
                    
                    bbox_to_vis[obj_id] = bbox  # Store bounding box for visualization
                    mask_to_vis[obj_id] = mask_binary  # Store mask for visualization

                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color
                    img = cv2.addWeighted(output_frame, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 255, 2)
                    
                out.write(img)
                    
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except StopIteration:
                break

    # Release resources
    out.release()
    cv2.destroyAllWindows()
    del predictor, inference_state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    print(f"Video saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM2 video predictor on a video file or camera.")
    parser.add_argument("--video", type=str, default=None, help="Path to the video file. If not provided, the camera will be used.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use (default: 0). Ignored if --video is provided.")
    parser.add_argument("--log_mem", type=bool, default=False, help="Log GPU memory usage.")
    args = parser.parse_args()
    main(args)