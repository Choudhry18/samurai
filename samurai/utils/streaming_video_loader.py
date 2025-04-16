import os
import warnings
import threading
from queue import Queue
from typing import List, Tuple, Dict, Optional, Iterator, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import time

from samurai.utils.misc import _load_img_as_tensor


class StreamingVideoLoader:
    """
    Video frame loader that loads frames in chunks to support streaming video processing.
    Maintains a sliding window of frames to support SAM2 temporal processing requirements.
    """

    def __init__(
        self,
        frame_iterator: Iterator[np.ndarray],
        image_size: int,
        offload_to_cpu: bool = False,
        chunk_size: int = 30,
        temporal_context: int = 4,  # Should match SAM2's temporal context requirements
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        compute_device=torch.device("cuda"),
    ):
        """
        Initialize a streaming video loader.
        
        Args:
            frame_iterator: Iterator yielding video frames as numpy arrays
            image_size: Target size to resize frames
            offload_to_cpu: Whether to keep frames on CPU (True) or GPU (False)
            chunk_size: Number of frames to keep in memory at once
            temporal_context: Number of frames required for temporal context
            img_mean: Normalization mean
            img_std: Normalization std
            compute_device: Device for tensor computations
        """
        self.frame_iterator = frame_iterator
        self.image_size = image_size
        self.offload_to_cpu = offload_to_cpu
        self.chunk_size = max(chunk_size, 2 * temporal_context + 1)  # Ensure enough context
        self.temporal_context = temporal_context
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        self.compute_device = compute_device
        
        # Frame management
        self.frames = {}  # Dictionary to store frames by index
        self.current_chunk_start = 0
        self.next_frame_idx = 0
        self.total_frame_count = 0
        self.video_height = None
        self.video_width = None
        
        # Threading and synchronization
        self.frame_queue = Queue(maxsize=self.chunk_size)
        self.exception = None
        self.loader_thread = None
        self.is_loading = False
        self.end_of_stream = False
        
        # Load first frame to get dimensions
        self._load_first_frame()
        
        # Start background loading
        self._start_loader_thread()

    def _load_first_frame(self):
        """Load the first frame to get video dimensions and initialize the loader."""
        try:
            first_frame = next(self.frame_iterator)
            tensor_frame, height, width = self._convert_frame_to_tensor(first_frame)
            self.video_height = height
            self.video_width = width
            
            # Store first frame
            self.frames[0] = tensor_frame
            self.next_frame_idx = 1
            self.total_frame_count = 1
        except StopIteration:
            raise RuntimeError("No frames available in video stream")
        except Exception as e:
            raise RuntimeError(f"Error loading first frame: {e}")

    def _convert_frame_to_tensor(self, frame: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """Convert numpy frame to normalized tensor."""
        # Get original dimensions
        height, width = frame.shape[:2]
        
        # Resize frame
        resized_frame = torch.from_numpy(
            np.array(Image.fromarray(frame).resize((self.image_size, self.image_size)))
        ).permute(2, 0, 1)
        
        if resized_frame.dtype == torch.uint8:
            resized_frame = resized_frame.float() / 255.0
            
        # Normalize
        resized_frame -= self.img_mean
        resized_frame /= self.img_std
        
        # Move to device if needed
        if not self.offload_to_cpu:
            resized_frame = resized_frame.to(self.compute_device, non_blocking=True)
            
        return resized_frame, height, width

    def _start_loader_thread(self):
        """Start background thread to load frames."""
        if self.loader_thread is not None and self.loader_thread.is_alive():
            return
            
        self.is_loading = True
        self.loader_thread = threading.Thread(
            target=self._load_frames_worker, 
            daemon=True
        )
        self.loader_thread.start()

    def _load_frames_worker(self):
        """Worker thread that loads frames and adds them to the queue."""
        try:
            while self.is_loading:
                try:
                    frame = next(self.frame_iterator)
                    tensor_frame, _, _ = self._convert_frame_to_tensor(frame)
                    self.frame_queue.put((self.next_frame_idx, tensor_frame))
                    self.next_frame_idx += 1
                except StopIteration:
                    self.end_of_stream = True
                    break
        except Exception as e:
            self.exception = e
        finally:
            self.is_loading = False

    def _process_queue(self):
        """Process frames in the queue and add them to the frames dictionary."""
        if self.exception:
            raise RuntimeError("Error in frame loading thread") from self.exception
            
        # Add frames from queue to our dictionary
        try:
            while not self.frame_queue.empty():
                idx, frame = self.frame_queue.get_nowait()
                self.frames[idx] = frame
                self.total_frame_count = max(self.total_frame_count, idx + 1)
        except Exception:
            pass  # Queue may become empty between check and get

    def _manage_chunk(self, requested_idx):
        """
        Manage which frames to keep in memory based on the requested index.
        Implements sliding window approach to maintain frame chunks with proper context.
        """
        self._process_queue()
        
        # Determine if we need to slide the window
        if requested_idx >= self.current_chunk_start + self.chunk_size:
            # Calculate new chunk boundaries, ensuring we keep temporal context
            new_start = max(0, requested_idx - self.temporal_context)
            new_end = new_start + self.chunk_size
            
            # Remove frames outside the new window but preserve temporal context
            to_remove = [i for i in self.frames.keys() 
                         if i < new_start - self.temporal_context or 
                            i >= new_end + self.temporal_context]
            
            for idx in to_remove:
                del self.frames[idx]
                
            self.current_chunk_start = new_start

    def __getitem__(self, idx):
        """Get a specific frame by index."""
        if self.exception:
            raise RuntimeError("Error in frame loading thread") from self.exception
            
        # First check if the frame is already loaded
        if idx in self.frames:
            return self.frames[idx]
            
        # Process any newly loaded frames
        self._process_queue()
        
        # Check again after processing queue
        if idx in self.frames:
            self._manage_chunk(idx)
            return self.frames[idx]
            
        # If we still don't have the frame and reached end of stream, it's an error
        if self.end_of_stream and idx >= self.total_frame_count:
            raise IndexError(f"Frame index {idx} out of range (total: {self.total_frame_count})")
            
        # If the frame is in the future but within expected range, wait for it
        if idx < self.next_frame_idx:
            # Wait for loader thread to catch up
            for _ in range(100):  # Timeout after 100 tries
                self._process_queue()
                if idx in self.frames:
                    self._manage_chunk(idx)
                    return self.frames[idx]
                time.sleep(0.01)  # Short sleep to avoid busy waiting
                
        raise IndexError(f"Frame {idx} not available (current range: {min(self.frames.keys() or [0])} to {max(self.frames.keys() or [0])})")

    def __len__(self):
        """
        Return the number of frames currently known.
        Note: This will increase as more frames are loaded.
        """
        return self.total_frame_count

    def get_current_range(self):
        """Get the range of frame indices currently available."""
        if not self.frames:
            return (0, 0)
        return (min(self.frames.keys()), max(self.frames.keys()) + 1)

    def stop(self):
        """Stop the loader thread."""
        self.is_loading = False
        if self.loader_thread and self.loader_thread.is_alive():
            self.loader_thread.join(timeout=1.0)


def track_streaming_video(predictor, frame_iterator, image_size=1024, device="cuda"):
    """
    Process a streaming video with SAMURAI.
    
    Args:
        predictor: SAM2VideoPredictor instance
        frame_iterator: Iterator yielding video frames as numpy arrays
        image_size: Size for resizing frames
        device: Compute device
    
    Returns:
        Generator yielding (frame_idx, object_ids, masks) for each processed frame
    """
    # 1. Set up the streaming video loader
    loader = StreamingVideoLoader(
        frame_iterator=frame_iterator,
        image_size=image_size,
        offload_to_cpu=device=="cpu",
        chunk_size=30,  # Adjust based on memory constraints
        temporal_context=4,  # Should match SAMURAI's requirements
        compute_device=torch.device(device)
    )
    
    # 2. Initialize the state with just the first frame
    # We'll create a simplified init_state that works with streaming input
    inference_state = {
        "device": torch.device(device),
        "storage_device": torch.device("cpu") if device=="cuda" else torch.device(device),
        "frames": loader,  # Use the loader instead of pre-loaded frames
        "video_height": loader.video_height,
        "video_width": loader.video_width,
        "num_frames": 1,  # Start with just 1 frame, will increase dynamically
        "cached_features": {},
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
        "frames_already_tracked": set(),
        "tracking_has_started": False,
        "obj_id_to_idx": {},
        "obj_idx_to_id": {},
        "constants": {"maskmem_pos_enc": None},
    }
    
    # 3. Process frames as they become available
    processed_frames = 0
    
    # Yield the output for each new frame as it's processed
    while True:
        try:
            # Update the number of frames we know about
            current_frame_count = len(loader)
            inference_state["num_frames"] = current_frame_count
            
            # Process any new frames
            while processed_frames < current_frame_count:
                # First frame needs a bbox or points
                if processed_frames == 0:
                    # Wait for user to provide bbox for first frame
                    yield processed_frames, [], None, "NEED_BBOX", inference_state
                    # At this point, caller should have called add_new_points_or_box
                    
                    # After user provides bbox, run inference for first frame
                    _ = predictor.propagate_in_video_preflight(inference_state)
                    frame_idx = 0
                    output_dict = inference_state["output_dict"]
                    
                    # Get frame data for first frame
                    feature_data = predictor._get_image_feature(
                        inference_state, 
                        frame_idx, 
                        batch_size=len(inference_state["obj_idx_to_id"])
                    )
                    
                    # Run inference on first frame
                    current_out, pred_masks = predictor._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=len(inference_state["obj_idx_to_id"]),
                        is_init_cond_frame=True,
                        point_inputs=None,  # We've already provided a bbox
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True
                    )
                    
                    # Get object IDs
                    object_ids = [
                        predictor._obj_idx_to_id(inference_state, obj_idx)
                        for obj_idx in range(len(inference_state["obj_idx_to_id"]))
                    ]
                    
                    # Get resized masks for output
                    video_res_masks = predictor._get_orig_video_res_output(
                        inference_state, 
                        pred_masks
                    )
                    
                    # Add to tracked frames
                    inference_state["frames_already_tracked"].add(frame_idx)
                    processed_frames += 1
                    
                    # Yield result for first frame
                    yield frame_idx, object_ids, video_res_masks
                
                else:
                    # Process subsequent frames
                    frame_idx = processed_frames
                    
                    # Skip if already processed
                    if frame_idx in inference_state["frames_already_tracked"]:
                        processed_frames += 1
                        continue
                    
                    # Run inference on this frame
                    batch_size = len(inference_state["obj_idx_to_id"])
                    output_dict = inference_state["output_dict"]
                    
                    current_out, pred_masks = predictor._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True
                    )
                    
                    # Add output to per-object storage
                    predictor._add_output_per_object(
                        inference_state, 
                        frame_idx, 
                        current_out, 
                        "non_cond_frame_outputs"
                    )
                    
                    # Get object IDs
                    object_ids = [
                        predictor._obj_idx_to_id(inference_state, obj_idx)
                        for obj_idx in range(batch_size)
                    ]
                    
                    # Get resized masks
                    video_res_masks = predictor._get_orig_video_res_output(
                        inference_state, 
                        pred_masks
                    )
                    
                    # Mark as processed
                    inference_state["frames_already_tracked"].add(frame_idx)
                    processed_frames += 1
                    
                    # Yield result
                    yield frame_idx, object_ids, video_res_masks
            
            # If we've reached the end of the stream, exit
            if loader.end_of_stream:
                break
                
            # Wait briefly for new frames
            import time
            time.sleep(0.01)
            
        except IndexError:
            # This happens when we try to access frames that aren't ready yet
            # Wait briefly for new frames
            import time
            time.sleep(0.1)