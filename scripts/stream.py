import cv2
from sam2.build_sam import build_sam2_video_predictor_hf
import numpy as np
import os
import torch

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

    # Define color for visualization
    color = [(255, 0, 0)]

    
    predictor.add_new_points_or_box(inference_state, box=bbox, frame_idx=0, obj_id=0)

    
    # Create a list to store frames so we can access previous frames
    frame_idx = 0
    mask_check = True

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while True:
            try:

                new_frame = next(frame_gen)
                frame_idx, object_ids, masks = predictor.propagate_streaming(inference_state, new_frame)

                
                # Create a copy for visualization
                output_frame = new_frame.copy()
                mask_to_vis = {}
                bbox_to_vis = {}

                # Visualize the masks
                for obj_id, mask in zip(object_ids, masks):  # masks[1] has the video_res_masks
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
                    if mask_check:
                        print(mask)
                        mask_check = False

                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] =  color
                    img = cv2.addWeighted(output_frame, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 255, 2)
                    
                    # mask_overlay = np.zeros_like(output_frame)
                    # mask_overlay[mask_binary] = [0, 0, 255]  # Red color for the mask
                    
                    # # Blend the mask with the original frame
                    # output_frame = cv2.addWeighted(output_frame, 1.0, mask_overlay, 0.5, 0)
                out.write(img)
                    
                # Draw the bounding box if we have one and this is frame 0
                # if frame_idx == 0 and bbox is not None:
                #     cv2.rectangle(
                #         output_frame, 
                #         (bbox[0], bbox[1]), 
                #         (bbox[2], bbox[3]), 
                #         (0, 255, 0), 
                #         2
                #     )
                
                # Display the frame and save to video
                # cv2.imshow("SAMURAI Tracking", output_frame)
                # out.write(output_frame)
                
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