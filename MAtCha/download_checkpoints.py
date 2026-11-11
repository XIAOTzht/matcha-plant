import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the pretrained models weights for the running the full MAtCha pipeline'
    )
    parser.add_argument('--depth_encoder', type=str, default='large', 
        help="Depth encoder to download. Can be 'small', 'base', 'large', or 'giant'."
    )
    args = parser.parse_args()

    encoder_key = {
        'small': 'vits',
        'base': 'vitb',
        'large': 'vitl',
        'giant': 'vitg'
    }
    
    # Download the DepthAnythingV2 checkpoint
    print(f"\n[INFO] Downloading the DepthAnythingV2 {args.depth_encoder.capitalize()} checkpoint...")
    os.system("mkdir -p ./Depth-Anything-V2/checkpoints/")
    os.system(f"wget https://huggingface.co/depth-anything/Depth-Anything-V2-{args.depth_encoder.capitalize()}/resolve/main/depth_anything_v2_{encoder_key[args.depth_encoder]}.pth -P ./Depth-Anything-V2/checkpoints/")
    print("[INFO] DepthAnythingV2 checkpoint downloaded.")
    
    # Download the MASt3R-SfM checkpoint
    print(f"\n[INFO] Downloading the MASt3R-SfM checkpoint...")
    os.system("mkdir -p ./mast3r/checkpoints/")
    os.system(f"wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P ./mast3r/checkpoints/")
    print("[INFO] MASt3R-SfM checkpoint downloaded.")

    # Download the MASt3R-SfM retrieval checkpoint
    print(f"\n[INFO] Downloading the MASt3R-SfM retrieval checkpoint...")
    os.system(f"wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P ./mast3r/checkpoints/")
    os.system(f"wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P ./mast3r/checkpoints/")
    print("[INFO] MASt3R-SfM retrieval checkpoint downloaded.")
