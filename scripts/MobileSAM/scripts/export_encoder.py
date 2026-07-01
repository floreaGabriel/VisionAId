"""
Export MobileSAM Image Encoder to ONNX for Android deployment.

This script exports ONLY the image encoder part of MobileSAM, which:
- Takes an image (3, 1024, 1024) as input
- Outputs image embeddings (256, 64, 64)

The decoder is already exported separately (mobile_sam.onnx).
"""

import torch
import warnings
import argparse

from mobile_sam import sam_model_registry

parser = argparse.ArgumentParser(
    description="Export MobileSAM image encoder to ONNX"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="../weights/mobile_sam.pt",
    help="Path to MobileSAM checkpoint"
)

parser.add_argument(
    "--output",
    type=str,
    default="mobile_sam_encoder.onnx",
    help="Output ONNX file name"
)

parser.add_argument(
    "--model-type",
    type=str,
    default="vit_t",
    help="Model type (vit_t for MobileSAM)"
)

parser.add_argument(
    "--opset",
    type=int,
    default=14,
    help="ONNX opset version"
)


class ImageEncoderONNX(torch.nn.Module):
    """
    Wrapper for MobileSAM image encoder for ONNX export.
    """
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        
    def forward(self, image):
        """
        Args:
            image: [1, 3, 1024, 1024] normalized image
        Returns:
            image_embeddings: [1, 256, 64, 64] embeddings
        """
        return self.image_encoder(image)


def export_encoder(
    checkpoint: str,
    output: str,
    model_type: str = "vit_t",
    opset: int = 14
):
    print(f"Loading MobileSAM model from {checkpoint}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.eval()
    
    # Create encoder wrapper
    encoder_model = ImageEncoderONNX(sam)
    encoder_model.eval()
    
    # Dummy input: [1, 3, 1024, 1024]
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    
    # Test forward pass
    print("Testing encoder forward pass...")
    with torch.no_grad():
        output_embeddings = encoder_model(dummy_input)
    print(f"Output shape: {output_embeddings.shape}")  # Should be [1, 256, 64, 64]
    
    # Export to ONNX
    print(f"Exporting encoder to {output}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        torch.onnx.export(
            encoder_model,
            dummy_input,
            output,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["image_embeddings"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "image_embeddings": {0: "batch_size"}
            }
        )
    
    print(f"✅ Encoder exported successfully to {output}")
    
    # Verify with ONNX Runtime
    try:
        import onnxruntime
        print("Verifying with ONNX Runtime...")
        
        ort_session = onnxruntime.InferenceSession(
            output,
            providers=["CPUExecutionProvider"]
        )
        
        # Test inference
        ort_inputs = {"image": dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✅ ONNX Runtime verification successful!")
        print(f"   Output shape: {ort_outputs[0].shape}")
        
        # Compare with PyTorch output
        max_diff = abs(output_embeddings.numpy() - ort_outputs[0]).max()
        print(f"   Max difference from PyTorch: {max_diff}")
        
    except ImportError:
        print("⚠️ ONNX Runtime not available for verification")
    except Exception as e:
        print(f"⚠️ ONNX Runtime verification failed: {e}")


if __name__ == "__main__":
    args = parser.parse_args()
    
    export_encoder(
        checkpoint=args.checkpoint,
        output=args.output,
        model_type=args.model_type,
        opset=args.opset
    )
    
    print("\n" + "="*60)
    print("📦 Next steps:")
    print("1. Copy the generated ONNX file to Android assets:")
    print(f"   cp {args.output} ../../app/src/main/assets/")
    print("2. The decoder is already at: app/src/main/assets/mobile_sam.onnx")
    print("3. Update MobileSAMSegmenter.kt to use both models")
    print("="*60)
