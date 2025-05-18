import os
from PIL import Image
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch

def generate_semantic_maps(input_dir, output_dir, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512").to(device)
    model.eval()

    img_names = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    for img_name in tqdm(img_names, desc=f"Processing {input_dir}"):
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"].to(device))
            seg = outputs.logits.argmax(dim=1)[0].cpu().numpy().astype('uint8')
        seg_img = Image.fromarray(seg)
        seg_img.save(os.path.join(output_dir, os.path.splitext(img_name)[0] + ".png"))

if __name__ == "__main__":
    data_root = r"D:\GIT\Datasets\BDD100k\100k_day2night"
    # 生成 trainS（train_A 的分割图）
    generate_semantic_maps(os.path.join(data_root, "train_A"), os.path.join(data_root, "trainS"))
    # 生成 trainT（train_B 的分割图）
    generate_semantic_maps(os.path.join(data_root, "train_B"), os.path.join(data_root, "trainT"))  