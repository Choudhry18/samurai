import cv2
from sam2.build_sam import build_sam2_video_predictor_hf
import numpy as np
import os

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


def main():
    # 1. Initialize model
    predictor = build_sam2_video_predictor_hf("facebook/sam2.1-hiera-base-plus", device="cuda:0")
    
    # 2. Setup streaming source
    video_source = "test_video.mp4"  # Path to video file
    if not os.path.exists(video_source):
        print(f"Error: Video file not found at {os.path.abspath(video_source)}")
        return
    frame_gen = frame_generator(video_source)
    
    # Get video properties for output file
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create video writer
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    first_frame = next(frame_gen)
    inference_state = predictor.init_streaming_state(first_frame)

    bbox = (300, 100, 50, 200)  # (x1, y1, x2, y2)
    predictor.add_new_points_or_box(inference_state, box=bbox, frame_idx=0, obj_id=0)

    
    # Create a list to store frames so we can access previous frames
    frames_buffer = []
    frame_idx = 0

    while True:
        try:

            new_frame = next(frame_gen)
            frame_idx, object_ids, masks = predictor.propagate_streaming(inference_state, new_frame)
            frame = frames_buffer[frame_idx]
            
            # Create a copy for visualization
            output_frame = frame.copy()
            
            # Visualize the masks
            for obj_id, mask in zip(object_ids, masks[1]):  # masks[1] has the video_res_masks
                mask_binary = mask.cpu().numpy() > 0
                
                # Create a colored mask overlay
                mask_overlay = np.zeros_like(output_frame)
                mask_overlay[mask_binary] = [0, 0, 255]  # Red color for the mask
                
                # Blend the mask with the original frame
                output_frame = cv2.addWeighted(output_frame, 1.0, mask_overlay, 0.5, 0)
                
            # Draw the bounding box if we have one and this is frame 0
            if frame_idx == 0 and bbox is not None:
                cv2.rectangle(
                    output_frame, 
                    (bbox[0], bbox[1]), 
                    (bbox[2], bbox[3]), 
                    (0, 255, 0), 
                    2
                )
            
            # Display the frame and save to video
            cv2.imshow("SAMURAI Tracking", output_frame)
            out.write(output_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except StopIteration:
            break

    # Release resources
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()