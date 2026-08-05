import cv2
import concurrent.futures
import os
import json
from tqdm import tqdm

output_dir = "/network_space/storage43/huzhuofan/Datasets/anet/sampled/"

def process_segment(video_path, start_sec, end_sec, idd):
    """
    Opens a video, seeks to the specified time segment,
    and uniformly samples 5 frames from that segment, saving them as JPGs.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If FPS is zero, we cannot proceed.
    if fps == 0:
        print(f"Failed to get FPS for video: {video_path}. Cannot process segment.")
        cap.release()
        return

    # Calculate the start and end frame numbers
    start_frame = min(total_frames - 1, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))
    
    # --- REVISED LOGIC: Uniformly sample 5 frames from the segment ---
    segment_length = end_frame - start_frame
    
    # If segment is too short (less than 5 frames), adjust sampling
    if segment_length < 4:
        # If segment is very short, just sample what we can
        frame_positions = list(range(start_frame, end_frame + 1))
        # Pad with the last frame if needed to get 5 frames
        while len(frame_positions) < 5:
            frame_positions.append(frame_positions[-1])
    else:
        # Uniformly sample 5 frames across the segment
        frame_positions = []
        for i in range(5):
            pos = start_frame + int(i * segment_length / 4)
            pos = min(total_frames - 1, max(0, pos))
            frame_positions.append(pos)

    # Extract and save the 5 frames
    for idx, frame_pos in enumerate(frame_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        
        if ret:
            # Save each frame with index suffix
            frame_path = f"{output_dir}{idd}_{idx:02d}.jpg"
            cv2.imwrite(frame_path, frame)
        else:
            print(f"Failed to extract frame at position {frame_pos} from {video_path}")
    
    cap.release()

def main():
    anno_file = "/network_space/storage43/huzhuofan/snag/no_dup.json"
    video_folder = "/network_space/storage43/huzhuofan/Datasets/anet/anet_6fps_224"
    with open(anno_file, "r") as f:
        data = json.load(f)
    
    origin_file = "/network_space/storage43/huzhuofan/MQ/anet_1.3.json"
    with open(origin_file, "r") as f:
        origin = json.load(f)

    tasks = []
    
    origin_data_splits = {'train': origin.get('train', {}), 'val': origin.get('val_1', {}), 'test': origin.get('val_2', {})}
    
    for split in ['train', 'val', 'test']:
        origin_vid_info = origin_data_splits[split]
        for key, vid_info in data[split].items():
            for item in vid_info['annotations']:
                id1, id2 = vid_info['video_id1'], vid_info['video_id2']
                video_id = id1 if item['location'] == 'video1' else id2
                query = item['sentence']
                start, end = None, None
                
                # Find the corresponding segment in the original annotations
                if video_id in origin_vid_info:
                    for item2 in origin_vid_info[video_id]['annotations']:
                        if query == item2['sentence']:
                            start, end = item2['segment']
                            break
                
                if start is None:
                    print(f"Can't find query_id {item['query_id']} with sentence '{query}' in original annotations for video {video_id}")
                    continue
                
                # Check for video file existence (.mp4 or .mkv)
                pattern_mp4 = os.path.join(video_folder, f"v_{video_id}.mp4")
                pattern_mkv = os.path.join(video_folder, f"v_{video_id}.mkv")
                
                video_path = None
                if os.path.exists(pattern_mp4):
                    video_path = pattern_mp4
                elif os.path.exists(pattern_mkv):
                    video_path = pattern_mkv
                
                if not video_path:
                    # Original code's video ID might not have the "v_" prefix, let's check that too
                    pattern_mp4_no_prefix = os.path.join(video_folder, f"{video_id}.mp4")
                    pattern_mkv_no_prefix = os.path.join(video_folder, f"{video_id}.mkv")
                    if os.path.exists(pattern_mp4_no_prefix):
                        video_path = pattern_mp4_no_prefix
                    elif os.path.exists(pattern_mkv_no_prefix):
                        video_path = pattern_mkv_no_prefix
                    else:
                        print(f"Video file for {video_id} not found.")
                        continue

                idd = item['query_id']
                tasks.append((video_path, start, end, idd))

    # Using a ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Create future tasks
        futures = [executor.submit(process_segment, *task) for task in tasks]
        
        # Use tqdm to show a progress bar
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Videos"):
            pass

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    main()