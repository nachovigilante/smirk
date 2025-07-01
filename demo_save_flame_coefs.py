import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
import argparse
import os
from utils.mediapipe_utils import run_mediapipe


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract FLAME coefficients from video and save to NPZ file')

    parser.add_argument('--input_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='flame_coefficients.npz', help='Path to save the NPZ file with FLAME coefficients')
    parser.add_argument('--skip_frames', type=int, default=1, help='Process every Nth frame (default: 1, process all frames)')

    args = parser.parse_args()

    image_size = 224
    
    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    # Initialize FLAME model
    flame = FLAME().to(args.device)

    # Open video file
    cap = cv2.VideoCapture(args.input_path)
    
    if not cap.isOpened():
        print(f'Error opening video file: {args.input_path}')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames at {video_fps} FPS")
    print(f"Processing every {args.skip_frames} frame(s)...")
    
    # Storage for all coefficients
    all_shape_params = []
    all_expression_params = []
    all_pose_params = []
    all_jaw_params = []
    all_eyelid_params = []
    all_cam_params = []
    all_frame_numbers = []
    all_timestamps = []
    
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, image = cap.read()
        
        if not ret:
            break
            
        # Skip frames if specified
        if frame_count % args.skip_frames != 0:
            frame_count += 1
            continue
            
        print(f"Processing frame {frame_count + 1}/{total_frames} (processed: {processed_count + 1})")
        
        timestamp = frame_count / video_fps if video_fps > 0 else frame_count

        kpt_mediapipe = run_mediapipe(image)

        # crop face if needed
        if args.crop:
            if kpt_mediapipe is None:
                print(f'Could not find landmarks for frame {frame_count + 1} using mediapipe and cannot crop the face. Skipping...')
                frame_count += 1
                continue
            
            kpt_mediapipe = kpt_mediapipe[..., :2]

            tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=image_size)
            
            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
        else:
            cropped_image = image

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_image = cropped_image.to(args.device)

        # Extract FLAME coefficients
        with torch.no_grad():
            outputs = smirk_encoder(cropped_image)
            
            # Store coefficients
            all_shape_params.append(outputs['shape_params'].cpu().numpy())
            all_expression_params.append(outputs['expression_params'].cpu().numpy())
            all_pose_params.append(outputs['pose_params'].cpu().numpy())
            all_jaw_params.append(outputs['jaw_params'].cpu().numpy())
            all_eyelid_params.append(outputs['eyelid_params'].cpu().numpy())
            all_cam_params.append(outputs['cam'].cpu().numpy())
            all_frame_numbers.append(frame_count)
            all_timestamps.append(timestamp)
            
        frame_count += 1
        processed_count += 1

    cap.release()

    if len(all_shape_params) == 0:
        print("No frames were processed successfully. Exiting...")
        exit()

    # Convert lists to numpy arrays
    all_shape_params = np.concatenate(all_shape_params, axis=0)
    all_expression_params = np.concatenate(all_expression_params, axis=0)
    all_pose_params = np.concatenate(all_pose_params, axis=0)
    all_jaw_params = np.concatenate(all_jaw_params, axis=0)
    all_eyelid_params = np.concatenate(all_eyelid_params, axis=0)
    all_cam_params = np.concatenate(all_cam_params, axis=0)

    # Save to NPZ file in format compatible with load_flame_params (Artalk format)
    np.savez(args.out_path,
             exp=all_expression_params,           # expression parameters
             gpose=all_pose_params,               # global pose parameters  
             jaw=all_jaw_params,                  # jaw pose parameters
             shape_params=all_shape_params,       # keep original for compatibility
             eyelid_params=all_eyelid_params,     # keep original for compatibility
             cam_params=all_cam_params,           # keep original for compatibility
             frame_numbers=np.array(all_frame_numbers),
             timestamps=np.array(all_timestamps),
             video_fps=video_fps,
             video_path=args.input_path)

    print(f"FLAME coefficients saved to: {args.out_path}")
    print(f"Shape parameters: {all_shape_params.shape}")
    print(f"Expression parameters (exp): {all_expression_params.shape}")
    print(f"Global pose parameters (gpose): {all_pose_params.shape}")
    print(f"Jaw parameters (jaw): {all_jaw_params.shape}")
    print(f"Eyelid parameters: {all_eyelid_params.shape}")
    print(f"Camera parameters: {all_cam_params.shape}")
    print(f"Processed {processed_count} frames from {total_frames} total frames")
    print(f"Video FPS: {video_fps}")
    print(f"Frame skip: {args.skip_frames}")
    print("File format: Compatible with load_flame_params() function (Artalk format)")
