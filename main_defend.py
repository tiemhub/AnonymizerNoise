import os
import argparse
from opt_anon import PoisonGeneration
from PIL import Image

def main():
    """
    Main function to parse arguments and run the PoisonGeneration process.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Generate poisoned images with specified parameters.")
    
    parser.add_argument("--device", type=str, default="0", 
                        help="The number of device to use for processing (e.g., '0', '1,2').")
    parser.add_argument("--eps", type=float, default=0.02,
                        help="The epsilon value for the poison generation process.")
    parser.add_argument("--disrupt", type=bool, default=False,
                        help="A boolean flag to determine whether to use loss_disrupt")
    parser.add_argument("--fr", type=bool, default=False,
                        help="A boolean flag to determine whether to use loss_fr")
    parser.add_argument("--HF_TOKEN", type=str, default=None,
                        help="Token for face recognition model in hugging face")
    parser.add_argument("--id_folder", type=str, default="./data/id",
                        help="The path of image folder to protect")
    parser.add_argument("--anon_folder", type=str, default="./data/anon",
                        help="The path of image folder contatin anonymized images")
    parser.add_argument("--output_folder", type=str, default="./data/output")

    # Parse the arguments from the command line
    args = parser.parse_args()
    
    # Set CUDA device visibility based on the provided device argument
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # Initialize the PoisonGeneration class with the parsed arguments
    generator = PoisonGeneration(device="cuda", eps=args.eps, disrupt=args.disrupt)
    
    # Define folder paths
    id_folder = args.id_folder
    anon_folder = args.anon_folder
    
    # Construct the output folder name based on the contrast value
    output_folder = args.output_folder
    
    # Run the image generation process
    generator.generate_all(id_folder=id_folder, anon_folder=anon_folder, output_folder=output_folder)
    print(f"Poison generation complete. Images saved to: {output_folder}")

if __name__ == "__main__":
    main()