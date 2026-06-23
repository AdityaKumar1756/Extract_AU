#!/usr/bin/env python
"""
Extract Action Units (AUs) and Emotions from Video at 30fps
This script processes videos of any length and extracts AU values and emotions for each frame.
Outputs: CSV and NPY files with AU values (20 AUs) and emotions (7 emotions).

The script can handle videos of any duration - from seconds to hours.
Processing time is approximately 1-2 seconds per frame on CPU.
"""

import os
import sys
import numpy as np
import argparse
import cv2
import tempfile
from tqdm import tqdm

# Ensure local py-feat source is available without needing it installed as a package
PROJECT_ROOT = os.path.dirname(__file__)
PYFEAT_SRC = os.path.join(PROJECT_ROOT, "py-feat")
if PYFEAT_SRC not in sys.path and os.path.isdir(PYFEAT_SRC):
    sys.path.insert(0, PYFEAT_SRC)

from feat import Detector


def extract_au_emotions(video_path, output_prefix=None, target_fps=30, rotate_frames=False):
    """
    Extract Action Units and Emotions from video at specified fps.
    
    Works with videos of any length (seconds to hours).
    
    Parameters:
    -----------
    video_path : str
        Path to input video file (any duration)
    output_prefix : str, optional
        Prefix for output files. If None, uses video filename
    target_fps : int, optional
        Target frames per second for processing (default: 30)
    rotate_frames : bool, optional
        Whether to rotate frames 90° clockwise before processing (default: False)
        
    Returns:
    --------
    dict : Dictionary containing AU data, emotion data, and full predictions
    """
    
    print("=" * 70)
    print("AU and Emotion Extraction from Video")
    print("=" * 70)
    
    # Validate input video
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set output prefix
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"\n[OK] Created results directory: {results_dir}")
    
    # Update output prefix to include results directory
    output_prefix = os.path.join(results_dir, output_prefix)
    
    # Initialize detector
    print("\n1. Initializing Py-Feat Detector (using CPU)...")
    detector = Detector(device='cpu')
    print(f"   [OK] Detector ready")
    print(f"   - Face model: {detector.info['face_model']}")
    print(f"   - AU model: {detector.info['au_model']}")
    print(f"   - Emotion model: {detector.info['emotion_model']}")
    
    # Get video info
    print(f"\n2. Processing video: {video_path}")
    
    # Calculate skip_frames based on target fps
    # Note: skip_frames=0 causes a bug in py-feat, use skip_frames=1 for every frame
    # skip_frames=1 means process every frame
    # skip_frames=2 means process every other frame, etc.
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0
    cap.release()
    
    # Calculate skip_frames to achieve target fps
    if video_fps >= target_fps:
        skip_frames = max(1, int(round(video_fps / target_fps)))
    else:
        skip_frames = 1  # Video is already slower than target fps
    
    print(f"   - Video FPS: {video_fps:.2f}")
    print(f"   - Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"   - Total frames in video: {frame_count}")
    print(f"   - Target FPS: {target_fps}")
    print(f"   - Skip frames: {skip_frames} (process every {skip_frames} frame(s))")
    expected_output_frames = frame_count // skip_frames
    print(f"   - Expected output frames: ~{expected_output_frames}")
    if rotate_frames:
        print(f"   - Frames will be rotated 90° clockwise before processing")
    print("   Note: This may take several minutes for long videos...")
    
    # Rotate video if requested
    video_to_process = video_path
    temp_video_path = None
    
    if rotate_frames:
        print(f"\n2.5. Rotating video frames 90° clockwise...")
        # Create temporary rotated video
        temp_fd, temp_video_path = tempfile.mkstemp(
            suffix='.mp4', dir=os.path.dirname(video_path) or '.'
        )
        os.close(temp_fd)

        # Open input video
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # After rotating 90° clockwise, dimensions swap
        new_width = height
        new_height = width

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_width, new_height))

        if not out.isOpened():
            cap.release()
            raise ValueError(f"Could not create temporary rotated video: {temp_video_path}")

        print(f"      Rotating {total_frames} frames...")
        pbar = tqdm(total=total_frames, desc="      Rotating frames", unit="frame")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Rotate 90 degrees clockwise
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Write rotated frame
            out.write(rotated_frame)

            frame_num += 1
            pbar.update(1)

        pbar.close()
        print(f"      Rotated {frame_num} frames successfully")

        # Release everything
        cap.release()
        out.release()

        video_to_process = temp_video_path
        print(f"      Using rotated video for processing")
    
    # Detect facial expressions in the video
    try:
        # Use the video-specific API available in this py-feat version
        video_prediction = detector.detect_video(
            video_to_process,
            skip_frames=skip_frames,
            face_detection_threshold=0.5,  # More lenient threshold
            output_size=None,              # Keep original frame size
        )
    except Exception as e:
        print(f"\n[ERROR] Error during video processing: {e}")
        raise
    finally:
        # Clean up temporary rotated video
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                if rotate_frames:
                    print(f"   [OK] Cleaned up temporary rotated video")
            except:
                pass  # Ignore cleanup errors
    
    print(f"\n3. Detection complete!")
    print(f"   - Processed {len(video_prediction)} frames")
    print(f"   - Total columns: {video_prediction.shape[1]}")
    
    # Get video metadata
    if len(video_prediction) > 0:
        first_row = video_prediction.iloc[0]
        frame_height = first_row.get('FrameHeight', 'N/A')
        frame_width = first_row.get('FrameWidth', 'N/A')
        print(f"   - Frame size: {frame_width}x{frame_height}")
        
        # Calculate actual fps from the data
        if 'approx_time' in video_prediction.columns:
            try:
                total_time = float(video_prediction['approx_time'].max())
                actual_fps = len(video_prediction) / total_time if total_time > 0 else 0
                print(f"   - Video duration: {total_time:.2f} seconds")
                print(f"   - Actual FPS: {actual_fps:.2f}")
            except (ValueError, TypeError):
                print(f"   - Video duration: N/A")
                print(f"   - Actual FPS: N/A")
    
    # Extract AU columns (all columns starting with 'AU')
    au_columns = [col for col in video_prediction.columns if col.startswith('AU')]
    print(f"\n4. Action Units detected: {len(au_columns)} AUs")
    print(f"   AUs: {', '.join(au_columns)}")
    
    # Extract emotion columns
    emotion_columns = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    available_emotions = [col for col in emotion_columns if col in video_prediction.columns]
    print(f"\n5. Emotions detected: {len(available_emotions)} emotions")
    print(f"   Emotions: {', '.join(available_emotions)}")
    
    # Extract AU data
    au_data = video_prediction[au_columns].values
    print(f"\n6. AU data shape: {au_data.shape}")
    print(f"   - Frames: {au_data.shape[0]}")
    print(f"   - AUs per frame: {au_data.shape[1]}")
    
    # Extract emotion data
    emotion_data = video_prediction[available_emotions].values
    print(f"\n7. Emotion data shape: {emotion_data.shape}")
    print(f"   - Frames: {emotion_data.shape[0]}")
    print(f"   - Emotions per frame: {emotion_data.shape[1]}")
    
    # Combine AU and emotion data
    combined_data = np.hstack([au_data, emotion_data])
    print(f"\n8. Combined data shape: {combined_data.shape}")
    print(f"   - Total features per frame: {combined_data.shape[1]} ({au_data.shape[1]} AUs + {emotion_data.shape[1]} emotions)")
    
    # Save outputs
    print("\n9. Saving outputs...")
    
    # Save full CSV with all columns
    csv_full_path = f"{output_prefix}_full_predictions.csv"
    video_prediction.to_csv(csv_full_path, index=False)
    print(f"   [OK] Full CSV saved: {csv_full_path}")
    
    # Save AU + Emotion CSV (compact)
    au_emotion_df = video_prediction[['frame', 'approx_time'] + au_columns + available_emotions].copy()
    csv_compact_path = f"{output_prefix}_au_emotions.csv"
    au_emotion_df.to_csv(csv_compact_path, index=False)
    print(f"   [OK] Compact CSV saved: {csv_compact_path}")
    
    # Save AU data as NPY
    au_npy_path = f"{output_prefix}_au_data.npy"
    np.save(au_npy_path, au_data)
    print(f"   [OK] AU data (NPY) saved: {au_npy_path}")
    print(f"     Shape: {au_data.shape}")
    
    # Save emotion data as NPY
    emotion_npy_path = f"{output_prefix}_emotion_data.npy"
    np.save(emotion_npy_path, emotion_data)
    print(f"   [OK] Emotion data (NPY) saved: {emotion_npy_path}")
    print(f"     Shape: {emotion_data.shape}")
    
    # Save combined AU + emotion data as NPY
    combined_npy_path = f"{output_prefix}_au_emotions.npy"
    np.save(combined_npy_path, combined_data)
    print(f"   [OK] Combined AU+Emotion data (NPY) saved: {combined_npy_path}")
    print(f"     Shape: {combined_data.shape}")
    
    # Save column names for reference
    column_names_path = f"{output_prefix}_column_names.txt"
    with open(column_names_path, 'w') as f:
        f.write("Action Units:\n")
        for i, col in enumerate(au_columns):
            f.write(f"  Column {i}: {col}\n")
        f.write(f"\nEmotions:\n")
        for i, col in enumerate(available_emotions):
            f.write(f"  Column {len(au_columns) + i}: {col}\n")
    print(f"   [OK] Column names saved: {column_names_path}")
    
    # Display summary statistics
    print("\n10. Summary Statistics:")
    print("\n    Action Units (mean values across all frames):")
    for col in au_columns[:10]:  # Show first 10 AUs
        mean_val = video_prediction[col].mean()
        std_val = video_prediction[col].std()
        print(f"      {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    if len(au_columns) > 10:
        print(f"      ... and {len(au_columns) - 10} more AUs")
    
    print("\n    Emotions (mean values across all frames):")
    for col in available_emotions:
        mean_val = video_prediction[col].mean()
        std_val = video_prediction[col].std()
        print(f"      {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Frame count summary
    actual_frames = len(video_prediction)
    print(f"\n11. Processing summary:")
    print(f"    - Total frames processed: {actual_frames}")
    print(f"    - Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"    - Average processing speed: {actual_frames / duration * skip_frames:.2f} fps")
    
    print("\n" + "=" * 70)
    print("[OK] Extraction complete!")
    print("=" * 70)
    
    print("\nOutput files created:")
    print(f"  1. {csv_full_path} - Full predictions with all columns")
    print(f"  2. {csv_compact_path} - Compact CSV with frame, time, AUs, emotions")
    print(f"  3. {au_npy_path} - NumPy array of AU values only")
    print(f"  4. {emotion_npy_path} - NumPy array of emotion values only")
    print(f"  5. {combined_npy_path} - NumPy array of combined AU+emotion values")
    print(f"  6. {column_names_path} - Column names reference")
    
    print("\nData format:")
    print(f"  - AU array shape: (num_frames={au_data.shape[0]}, num_aus={au_data.shape[1]})")
    print(f"  - Emotion array shape: (num_frames={emotion_data.shape[0]}, num_emotions={emotion_data.shape[1]})")
    print(f"  - Combined array shape: (num_frames={combined_data.shape[0]}, num_features={combined_data.shape[1]})")
    
    return {
        'au_data': au_data,
        'emotion_data': emotion_data,
        'combined_data': combined_data,
        'au_columns': au_columns,
        'emotion_columns': available_emotions,
        'predictions': video_prediction,
        'num_frames': actual_frames
    }


def main():
    """Main function to parse arguments and run extraction.

    This entry point now supports processing one or more videos in a single
    command. When multiple video paths are provided, each video is processed
    sequentially with its own output prefix based on the filename.
    """
    parser = argparse.ArgumentParser(
        description='Extract Action Units and Emotions from video(s) at a target FPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video with automatic output naming
  python extract_au_emotions.py input_video.mp4

  # Process a single video with custom output prefix
  python extract_au_emotions.py input_video.mp4 --output my_results

  # Process a single video at different fps
  python extract_au_emotions.py input_video.mp4 --fps 24

  # Process multiple videos in one command (each uses its own filename as prefix)
  python extract_au_emotions.py vid1.mp4 vid2.mp4 vid3.mp4 --fps 1
        """
    )
    
    parser.add_argument(
        'video_paths',
        nargs='+',
        type=str,
        help='One or more paths to input video file(s)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output prefix for generated files (default: video filename)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target frames per second for processing (default: 30)'
    )
    
    args = parser.parse_args()
    
    exit_code = 0

    # If multiple videos are provided and a custom output prefix was given,
    # we ignore the custom prefix and fall back to per-video filenames to
    # avoid collisions.
    use_custom_output = args.output is not None and len(args.video_paths) == 1
    if args.output is not None and len(args.video_paths) > 1:
        print(
            "[INFO] --output was provided but multiple video paths were given.\n"
            "       The custom output prefix will be ignored and each video\n"
            "       will use its own filename as the output prefix."
        )

    for idx, video_path in enumerate(args.video_paths, 1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(args.video_paths)}] Processing video: {video_path}")
        print("=" * 70)

        try:
            # When processing a single video and a custom output was given,
            # use that. Otherwise let extract_au_emotions derive the prefix
            # from the video filename.
            output_prefix = args.output if use_custom_output else None

            extract_au_emotions(
                video_path=video_path,
                output_prefix=output_prefix,
                target_fps=args.fps
            )
        except Exception as e:
            print(f"\n[ERROR] Error while processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            exit_code = 1

    # Show example loading code once at the end
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("Success! You can now load the data in Python (example):")
        print("=" * 70)
        print("""
import numpy as np
import pandas as pd

# Load NPY files
au_data = np.load('OUTPUT_au_data.npy')
emotion_data = np.load('OUTPUT_emotion_data.npy')
combined_data = np.load('OUTPUT_au_emotions.npy')

# Load CSV file
df = pd.read_csv('OUTPUT_au_emotions.csv')

# Access data
print(f"AU data shape: {au_data.shape}")
print(f"Emotion data shape: {emotion_data.shape}")
print(f"First frame AUs: {au_data[0]}")
print(f"First frame emotions: {emotion_data[0]}")
        """)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

