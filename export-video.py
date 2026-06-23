import sys
import os
import glob
import h5py
import cv2

if len(sys.argv) < 2:
    print("Usage: python export_video.py <path_to_h5_file_or_directory>")
    sys.exit(1)

input_path = sys.argv[1]
output_dir_name = 'output-new'

if not os.path.exists(input_path):
    print(f"Error: '{input_path}' was not found.")
    sys.exit(1)

if os.path.isdir(input_path):
    h5_files = sorted(glob.glob(os.path.join(input_path, '*.h5')))
    if not h5_files:
        print(f"No .h5 files found in '{input_path}'.")
        sys.exit(1)
else:
    h5_files = [input_path]


def process(input_file):
    base_dir = os.path.dirname(os.path.abspath(input_file))
    video_dir = os.path.join(base_dir, output_dir_name, 'videos')
    os.makedirs(video_dir, exist_ok=True)

    log_path = os.path.join(base_dir, output_dir_name, 'processed.txt')
    abs_input = os.path.abspath(input_file)

    if os.path.exists(log_path):
        with open(log_path, 'r') as log:
            processed = {line.strip() for line in log}
        if abs_input in processed:
            print(f"Skipping (already processed): {abs_input}")
            return

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    print(f"\nReading file: {input_file}")

    min_size = 1.7 * 1024 ** 3
    if os.path.getsize(input_file) < min_size:
        print(f"Skipping {input_file}: file is smaller than 1.7GB, likely truncated.")
        return

    try:
        h5_handle = h5py.File(input_file, 'r')
    except OSError as e:
        print(f"Error opening {input_file}: {e}\nSkipping (file may be corrupted or truncated).")
        return

    with h5_handle as f:
        for side in ('left', 'right'):
            key = f'rgb_{side}'
            video_path = os.path.join(video_dir, f"{base_name}_rgb_{side}.mp4")
            video_data = f[key]
            num_frames, height, width = video_data.shape[:3]

            print(f"Exporting {key}: {num_frames} frames at {width}x{height}...")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

            bytes_per_frame = height * width * 3
            chunk_size = max(1, (5 * 1024 ** 3) // bytes_per_frame)
            for start in range(0, num_frames, chunk_size):
                chunk = video_data[start:start + chunk_size]
                for frame in chunk:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"  {min(start + chunk_size, num_frames)} / {num_frames} frames...")

            out.release()
            print(f"Video saved to {video_path}")

    with open(log_path, 'a') as log:
        log.write(abs_input + '\n')


for h5_file in h5_files:
    process(h5_file)

print("\nAll done!")
