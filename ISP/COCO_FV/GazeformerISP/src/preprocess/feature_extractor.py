# --- Code Cũ (Không chạy) ---
# from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
# class ResNetCOCO(nn.Module): ...
# bbone = ResNetCOCO(device=device).to(device).eval()
# ...
# features = bbone(tensor_image).squeeze().detach().cpu()
# torch.save(features, join(target_path, f.replace('jpg', 'pth')))

# --- Code Mới (Tương thích) ---
# Hãy thay thế toàn bộ nội dung file bằng code này.
# Nó sử dụng Detectron2 để tải mô hình một cách chính xác.

import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Hàm trích xuất đặc trưng ảnh
def image_data(dataset_path, output_path, device='cuda:0', overwrite=False):
    # Kích thước ảnh đầu vào cho ResNet-50
    resize_dim = (800, 1066) # Kích thước chuẩn cho backbone
    transform = T.Compose([
        T.Resize(resize_dim),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Tải mô hình ResNet-50 FPN (phần body) đã huấn luyện trên COCO
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body.to(device)
    model.eval()

    print(f"Bắt đầu trích xuất đặc trưng từ: {dataset_path}")
    print(f"Lưu vào: {output_path}")

    # Tạo thư mục output nếu chưa có
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # Lấy danh sách các thư mục con (ví dụ: 'bottle', 'bowl', 'car'...)
    categories = [d for d in os.listdir(dataset_path) if isdir(join(dataset_path, d))]
    if not categories: # Nếu không có thư mục con (như OSIE)
        categories = [""] # Chạy 1 lần cho thư mục gốc

    for task in categories:
        print(f"Đang xử lý thư mục: '{task}'")
        src_path = join(dataset_path, task)
        target_path = join(output_path, task)

        if not os.path.exists(target_path):
            os.makedirs(target_path, exist_ok=True)
            
        files = [f for f in os.listdir(src_path) if isfile(join(src_path, f)) and f.endswith('.jpg')]
        
        for f in tqdm(files, desc=f"Task {task}"):
            target_file = join(target_path, f.replace('.jpg', '.pth'))
            if not overwrite and os.path.exists(target_file):
                continue

            try:
                PIL_image = PIL.Image.open(join(src_path, f)).convert("RGB")
                tensor_image = transform(PIL_image).unsqueeze(0).to(device) # Tạo batch size 1

                with torch.no_grad():
                    # Chạy mô hình và lấy đặc trưng từ lớp cuối cùng (layer4)
                    features = model(tensor_image)['res5'] # res5 là tên chuẩn cho layer4
                
                # Biến đổi kích thước về (Số_patches, Số_kênh)
                bs, ch, h, w = features.shape
                features_flat = features.view(bs, ch, -1).permute(0, 2, 1).squeeze(0) # (H*W, C)
                
                torch.save(features_flat.detach().cpu(), target_file)
            except Exception as e:
                print(f"Lỗi khi xử lý file {f}: {e}")

# Hàm trích xuất đặc trưng văn bản
def text_data(dataset_path, output_path, device='cuda:0', lm_model='sentence-transformers/stsb-roberta-base-v2'):
    # Lấy danh sách các task từ thư mục ảnh
    tasks = [d for d in os.listdir(dataset_path) if isdir(join(dataset_path, d))]
    
    # Thêm 'free-viewing' cho OSIE và COCO_FV
    if not tasks: # Trường hợp OSIE/COCO_FV
        tasks = ['free-viewing']
    else: # Trường hợp COCO_Search18
        # Chuyển tên thư mục 'potted_plant' thành 'potted plant'
        tasks = [t.replace('_', ' ') for t in tasks]

    print(f"Đang mã hóa các task: {tasks}")
    lm = SentenceTransformer(lm_model, device=device).eval()
    embed_dict = {}
    
    for task in tasks:
        # Key cho COCO_Search18 là 'person', 'car', v.v.
        key = task
        # Key cho OSIE/COCO_FV
        if task == 'free-viewing':
            prompt = 'You are allowed to view visual stimuli or scenes without any specific task or instruction.'
        else:
            prompt = task

        embed_dict[key] = lm.encode(prompt)

    # Đổi tên file cho rõ ràng
    save_filename = 'coco_search18_embeddings.npy' if 'person' in embed_dict else 'osie_coco_fv_embeddings.npy'
    
    with open(join(output_path, save_filename), 'wb') as f:
        np.save(f, embed_dict, allow_pickle=True)
    print(f"Đã lưu text embeddings vào: {join(output_path, save_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gazeformer Feature Extractor Utils', add_help=False)
    # Đường dẫn này sẽ được thay đổi khi chạy trên Kaggle
    parser.add_argument('--dataset_path', default= '../../../SE-Net/data/COCO_FV', type=str) 
    parser.add_argument('--output_path', default= 'src/data', type=str)
    parser.add_argument('--lm_model', default= 'sentence-transformers/stsb-roberta-base-v2', type=str)
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # Tạo thư mục output chính
    if not os.path.exists(join(args.output_path, 'image_features')):
        os.makedirs(join(args.output_path, 'image_features'), exist_ok=True)
        
    print("--- Bắt đầu trích xuất đặc trưng hình ảnh ---")
    image_data(dataset_path=args.dataset_path, output_path=join(args.output_path, 'image_features'), device=device, overwrite=True)
    
    print("\n--- Bắt đầu trích xuất đặc trưng văn bản ---")
    text_data(dataset_path=args.dataset_path, output_path=args.output_path, device=device, lm_model=args.lm_model)