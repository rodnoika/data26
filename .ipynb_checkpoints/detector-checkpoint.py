import cv2
import json
import os

def draw_detections_on_video(input_video_path, detections_data, output_video_path, line_thickness=2, font_scale=0.6):
    """
    Reads an input video, overlays object detection results onto each frame, 
    and saves the result to an output video.

    Parameters:
    - input_video_path (str): Path to the source video file (e.g., .mov or .mp4).
    - detections_data (dict or list): Detection results structured either by frame index 
      or by timestamps. Example structure expected:
      {
          "frames": {
              "0": [
                  {"bbox": [x1, y1, x2, y2], "class": "cashier", "confidence": 0.95},
                  {"bbox": [x1, y1, x2, y2], "class": "money", "confidence": 0.88}
              ],
              "30": [...]
          }
      }
      OR a list where index corresponds to frame number.
    - output_video_path (str): Path where the processed video will be saved.
    - line_thickness (int): Thickness of the bounding box lines.
    - font_scale (float): Scale of the text font.
    """
    
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or fps is None:
        fps = 30.0 # Default fallback

    # Define codec and create VideoWriter object
    # 'mp4v' or 'XVID' are widely compatible. For .mov or .mp4 output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    print(f"Processing video: {input_video_path} ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have detections for the current frame index
        # Supporting both dictionary formats ("frames" key or direct mapping)
        frame_key = str(frame_idx)
        detections = []
        
        if isinstance(detections_data, dict):
            frames_dict = detections_data.get("frames", detections_data)
            detections = frames_dict.get(frame_key, frames_dict.get(frame_idx, []))
        elif isinstance(detections_data, list) and frame_idx < len(detections_data):
            detections = detections_data[frame_idx]

        # Draw each detection on the frame
        for det in detections:
            # Expecting bbox in format [x1, y1, x2, y2] (pixel coordinates)
            bbox = det.get("bbox")
            class_name = det.get("class", "object")
            confidence = det.get("confidence", None)

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Choose color based on class or default to green
                color = (0, 255, 0) # BGR format: Green
                if "violation" in class_name.lower():
                    color = (0, 0, 255) # Red for potential violations

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

                # Prepare label text
                if confidence is not None:
                    label = f"{class_name}: {confidence:.2f}"
                else:
                    label = f"{class_name}"

                # Put label background and text for readability
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Ensure label doesn't go off the top of the frame
                y1_text = max(y1, text_height + 10)
                
                cv2.rectangle(
                    frame, 
                    (x1, y1_text - text_height - 10), 
                    (x1 + text_width + 10, y1_text + baseline), 
                    color, 
                    cv2.FILLED
                )
                
                cv2.putText(
                    frame, 
                    label, 
                    (x1 + 5, y1_text - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )

        # Optional: Render current timestamp / timecode on the frame
        current_time_sec = frame_idx / fps
        time_text = f"Time: {current_time_sec:.2f}s (Frame {frame_idx})"
        cv2.putText(
            frame, 
            time_text, 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )

        # Write the processed frame to the output video
        out.write(frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Successfully saved output video to: {output_video_path}")

if __name__ == "__main__":
    # Example usage:
    # 1. Load detection output from a JSON file (or use your pipeline's output directly)
    json_path = "detections_output.json"
    with open(json_path, "r") as f:
         detections_dict = json.load(f)
    
    # 2. Run the visualization function
    draw_detections_on_video(
         input_video_path="X-1.mov",
         detections_data=detections_dict,
         output_video_path="output_visualized.mp4"
    )
    pass