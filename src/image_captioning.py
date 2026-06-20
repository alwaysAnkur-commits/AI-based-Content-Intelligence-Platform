# src/image_captioning.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json, torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def caption_image(image_path: str) -> str:
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

def caption_batch(image_dir: str, output_path: str = "data/processed/image_captions.json") -> list:
    results = []
    image_paths = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
    for img_path in image_paths:
        try:
            caption = caption_image(str(img_path))
            results.append({"image_path": str(img_path), "image_name": img_path.name, "caption": caption})
            print(f"{img_path.name}: {caption}")
        except Exception as e:
            print(f"Failed on {img_path.name}: {e}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCaptioned {len(results)} images.")
    return results

if __name__ == "__main__":
    caption_batch("data/raw/extracted_images")