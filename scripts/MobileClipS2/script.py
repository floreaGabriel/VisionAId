import torch
import open_clip
from PIL import Image
import os
import sys

# Adaugă ml-mobileclip la path
sys.path.insert(0, 'ml-mobileclip')
from mobileclip.modules.common.mobileone import reparameterize_model

print("=" * 60)
print("MobileCLIP2-S2 → ONNX Export Script")
print("=" * 60)

# Calea către checkpoint-ul descărcat de pe HuggingFace
checkpoint_path = os.path.expanduser("~/.cache/huggingface/hub/models--apple--MobileCLIP2-S2/snapshots/72424e7025436db18f15c3eff6ee8c7c15ad4481/mobileclip2_s2.pt")

# Verifică dacă există și fișierul local
if os.path.exists("MobileCLIP2-S2.pt"):
    checkpoint_path = "MobileCLIP2-S2.pt"
    
print(f"📂 Loading checkpoint from: {checkpoint_path}")

# Creează modelul cu OpenCLIP
# Pentru MobileCLIP2-S2, trebuie image_mean și image_std custom
model, _, preprocess = open_clip.create_model_and_transforms(
    'MobileCLIP2-S2', 
    pretrained=checkpoint_path,
    image_mean=(0, 0, 0),
    image_std=(1, 1, 1)
)

print("✅ Model loaded!")

# FOARTE IMPORTANT: Reparametrizare pentru export
# Aceasta convertește operațiile dinamice în statice
print("🔧 Reparameterizing model for ONNX export...")
model.eval()
model = reparameterize_model(model)
print("✅ Model reparameterized!")

# Pregătește input pentru export
# MobileCLIP2-S2 folosește 256x256
dummy_input = torch.randn(1, 3, 256, 256)

# Export doar visual encoder
print("📤 Exporting visual encoder to ONNX...")
output_path = "mobileclip2_s2_visual.onnx"

with torch.no_grad():
    torch.onnx.export(
        model.visual,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=['pixel_values'],
        output_names=['embeddings'],
        dynamic_axes={'pixel_values': {0: 'batch'}}
    )

file_size = os.path.getsize(output_path) / 1024 / 1024
print("=" * 60)
print(f"✅ SUCCESS! Exported: {output_path}")
print(f"📊 File size: {file_size:.2f} MB")
print("=" * 60)
print(f"\n📋 Next steps:")
print(f"   1. Copy to assets: cp {output_path} ../../app/src/main/assets/")
print(f"   2. Update CLIPFeatureExtractor.kt to use new model")