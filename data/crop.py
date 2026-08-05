import json
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

def load(path):
    with open(path, "r") as f:
        return json.load(f)

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from tqdm import tqdm
import torch
from torchvision.ops import box_convert

# 全局模型变量，将在子进程中初始化
model = None

def init_worker(dino_config, weights_path):
    """初始化工作进程，加载模型"""
    global model
    print(f"Initializing worker process {os.getpid()}...")
    model = load_model(dino_config, weights_path)
    print(f"Model loaded in worker {os.getpid()}")

def crop(image_source, boxes):
    """裁剪图像中面积最大的边界框"""
    h, w, _ = image_source.shape
    if boxes.shape[0] == 0:
        return None
    
    # 将boxes缩放到像素坐标，并转为xyxy格式
    boxes_pixel = boxes * torch.tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # 计算每个box的面积，选择面积最大的box
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    max_idx = areas.argmax()
    x1, y1, x2, y2 = xyxy[max_idx]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    # 返回边界框信息和对应的图像
    return {
        "box": (x1, y1, x2, y2),
        "area": areas[max_idx],
        "image_source": image_source
    }

def process_query(item, image_dir, save_dir, box_threshold, text_threshold):
    """处理单个query_id"""
    idx = item['query_id']
    sentence = item['sentence']
    
    # 检查是否已处理
    save_path = os.path.join(save_dir, f"{idx}_crop.jpg")
    if os.path.exists(save_path):
        return f"Skipped {idx} (already processed)"
    
    # 尝试加载该query_id对应的所有帧 (0-4)
    frame_results = []
    fallback_image = None  # 用于保存备用图像（第一帧）
    
    for frame_idx in range(5):
        image_path = os.path.join(image_dir, f"{idx}_{frame_idx:02d}.jpg")
        
        # 如果文件不存在，跳过
        if not os.path.exists(image_path):
            continue
        
        try:
            image_source, image = load_image(image_path)
            
            # 保存第一帧作为备用（如果后续没有检测到有效框）
            if fallback_image is None:
                fallback_image = image_source
            
            # 进行预测
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=sentence,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # 处理当前帧的结果
            result = crop(image_source, boxes)
            if result:
                result["frame_idx"] = frame_idx
                frame_results.append(result)
        
        except Exception as e:
            # print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # 处理结果：优先使用检测框，其次使用原始图像
    if frame_results:
        # 选择所有帧中面积最大的边界框
        best_result = max(frame_results, key=lambda x: x["area"])
        x1, y1, x2, y2 = best_result["box"]
        crop_img = best_result["image_source"][y1:y2, x1:x2]
        cv2.imwrite(save_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        return f"Processed {idx} (crop from frame {best_result['frame_idx']})"
    elif fallback_image is not None:
        # 没有检测到有效框，保存原始图像
        cv2.imwrite(save_path, cv2.cvtColor(fallback_image, cv2.COLOR_RGB2BGR))
        return f"Processed {idx} (no boxes, used fallback image)"
    else:
        # 连一帧图像都没有找到
        return f"Failed {idx} (no images found)"

if __name__ == "__main__":
    # 配置参数
    DINO_CONFIG = "/network_space/storage43/huzhuofan/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "/network_space/storage43/huzhuofan/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    IMAGE_DIR = "/network_space/storage43/huzhuofan/Datasets/anet/sampled"
    SAVE_DIR = "/network_space/storage43/huzhuofan/Datasets/anet/crop"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    # 确保输出目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加载数据
    data = load("/network_space/storage43/huzhuofan/snag/no_dup.json")
    
    # 收集所有任务
    all_tasks = []
    for split in ['train', 'val', 'test']:
        for key, vid_info in data[split].items():
            for item in vid_info['annotations']:
                all_tasks.append(item)
    
    print(f"Total tasks to process: {len(all_tasks)}")
    
    # 设置进程数（根据GPU显存调整）
    num_processes = min(6, cpu_count())  # 通常8个进程对于单个GPU是安全的
    print(f"Using {num_processes} processes")
    
    # 创建进程池并初始化
    with Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(DINO_CONFIG, WEIGHTS_PATH)
    ) as pool:
        # 创建处理函数的部分应用
        worker_func = partial(
            process_query,
            image_dir=IMAGE_DIR,
            save_dir=SAVE_DIR,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        
        # 并行处理任务
        results = []
        for result in tqdm(
            pool.imap_unordered(worker_func, all_tasks),
            total=len(all_tasks),
            desc="Processing queries"
        ):
            results.append(result)
    
    # 打印摘要
    print("\nProcessing summary:")
    status_counts = {}
    for res in results:
        status = res.split(" ")[0]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        print(f"{status}: {count}")