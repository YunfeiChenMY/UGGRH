import os
import json
import time
import logging
import shutil
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import requests
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential


# --------------------------
# 1. 配置与日志
# --------------------------
class Config:
    # COCO数据集路径配置
    coco_root = ""
    coco_image_to_text_dir = os.path.join(coco_root, "coco_image_to_text")
    coco_text_to_image_dir = os.path.join(coco_root, "coco_text_to_image")
    
    # 输出目录配置
    output_dir = os.path.join(coco_root, "coco_generated_pairs")
    coco_img2text_images_dir = os.path.join(output_dir, "coco_img2text_images")
    coco_img2text_texts_dir = os.path.join(output_dir, "coco_img2text_texts")
    
    # 日志文件路径
    log_file = os.path.join(output_dir, "coco_img2text_processing_log.txt")
    
    # 进度保存和恢复相关配置
    progress_file = os.path.join(output_dir, "coco_img2text_progress.json")
    processed_samples_file = os.path.join(output_dir, "coco_img2text_processed_samples.json")
    
    # API配置
    img2text_api_url = ""
    api_key = ""
    
    # 处理配置
    max_workers = 2  # 线程数
    batch_size = 10  # 每批处理的样本数量
    
    # 文件名后缀，用于标识生成的数据
    generated_text_suffix = "_gen.txt"


# --------------------------
# 2. 初始化日志
# --------------------------
def init_logger():
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.coco_img2text_images_dir, exist_ok=True)
    os.makedirs(Config.coco_img2text_texts_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=Config.log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"  # 追加模式
    )
    
    # 控制台日志输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)


# --------------------------
# 3. 工具函数
# --------------------------
def get_coco_img2text_samples():
    """从COCO数据集中获取需要进行图生文处理的样本"""
    img2text_samples = []
    
    # 从 coco_image_to_text/images 文件夹中获取图片
    images_dir = os.path.join(Config.coco_image_to_text_dir, "images")
    
    if not os.path.exists(images_dir):
        logging.error(f"COCO图生文图片文件夹不存在：{images_dir}")
        return []
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.png', '.jpeg')
    image_files = []
    for f in os.listdir(images_dir):
        if f.lower().endswith(image_extensions):
            image_files.append(f)
    
    logging.info(f"在 {images_dir} 中找到 {len(image_files)} 个图片文件")
    
    # 构建样本列表，不需要匹配文本文件
    for img_file in image_files:
        # 从文件名提取ID（去掉扩展名）
        name, ext = os.path.splitext(img_file)
        file_id = name
        
        img2text_samples.append({
            "id": file_id,
            "image_path": os.path.join(images_dir, img_file)
        })
    
    logging.info(f"构建了 {len(img2text_samples)} 个COCO图生文样本")
    return img2text_samples


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_img2text_api_with_retry(image_path):
    """调用图生文API，带重试机制"""
    try:
        with open(image_path, 'rb') as img_file:
            image_bytes = img_file.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            payload = {"image": image_b64}
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{Config.img2text_api_url}/caption",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            response_json = response.json()
            generated_text = response_json.get("cmh_caption")  # 改为取 cmh_caption
            return generated_text
    except Exception as e:
        logging.warning(f"图生文重试失败（{image_path}）：{str(e)}")
        raise  # 触发重试


def save_progress(processed_samples):
    """保存处理进度"""
    with open(Config.processed_samples_file, 'w', encoding='utf-8') as f:
        json.dump(processed_samples, f, ensure_ascii=False, indent=2)
    logging.info(f"进度已保存：已处理 {len(processed_samples)} 个样本")


def load_progress():
    """加载处理进度"""
    if not os.path.exists(Config.processed_samples_file):
        return set()
    
    try:
        with open(Config.processed_samples_file, 'r', encoding='utf-8') as f:
            processed_samples = set(json.load(f))
        logging.info(f"加载进度：已处理 {len(processed_samples)} 个样本")
        return processed_samples
    except Exception as e:
        logging.error(f"加载进度失败：{str(e)}")
        return set()


def process_sample(sample):
    """处理单个样本"""
    sample_id = sample["id"]
    image_path = sample["image_path"]
    
    try:
        # 复制原始图片到输出目录
        output_image_path = os.path.join(Config.coco_img2text_images_dir, f"{sample_id}.jpg")
        shutil.copy2(image_path, output_image_path)
        
        # 调用图生文API
        generated_text = call_img2text_api_with_retry(image_path)
        
        if generated_text:
            # 保存生成的文本
            text_filename = f"{sample_id}.txt"
            with open(os.path.join(Config.coco_img2text_texts_dir, text_filename), 'w', encoding='utf-8') as f:
                f.write(generated_text)
            
            logging.info(f"样本 {sample_id} 图生文处理成功")
            return sample_id, "success", "img2text"
        else:
            # 如果生成失败，创建一个空文本文件
            text_filename = f"{sample_id}.txt"
            with open(os.path.join(Config.coco_img2text_texts_dir, text_filename), 'w', encoding='utf-8') as f:
                f.write("")
            logging.warning(f"样本 {sample_id} 图生文处理失败，创建空文本文件")
            return sample_id, "failed", "img2text_failed"
    
    except Exception as e:
        logging.error(f"样本 {sample_id} 处理失败：{str(e)}")
        return sample_id, "error", str(e)


def process_batch(batch_samples, batch_idx):
    """处理一批样本"""
    results = []
    
    with ThreadPoolExecutor(max_workers=Config.max_workers) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in batch_samples}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # 保存批次结果
    batch_log = os.path.join(Config.output_dir, f"coco_img2text_batch_{batch_idx}_results.json")
    with open(batch_log, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"批次 {batch_idx} 处理完成，结果保存至 {batch_log}")
    return results


# --------------------------
# 4. 主流程
# --------------------------
def process_coco_img2text_samples():
    """处理COCO图生文样本的主流程"""
    init_logger()
    
    # 获取需要进行图生文处理的样本
    img2text_samples = get_coco_img2text_samples()
    
    if len(img2text_samples) == 0:
        logging.error("没有找到COCO图生文样本，请检查文件夹是否存在")
        return
    
    # 加载已处理的样本
    processed_samples = load_progress()
    
    # 过滤出未处理的样本
    unprocessed_samples = [s for s in img2text_samples if s["id"] not in processed_samples]
    
    logging.info(f"总共需要处理 {len(img2text_samples)} 个样本，已处理 {len(processed_samples)} 个，待处理 {len(unprocessed_samples)} 个")
    
    if len(unprocessed_samples) == 0:
        logging.info("所有样本已处理完成！")
        return
    
    # 按批次处理
    batches = [
        unprocessed_samples[i:i + Config.batch_size]
        for i in range(0, len(unprocessed_samples), Config.batch_size)
    ]
    
    logging.info(f"共分为 {len(batches)} 个批次进行处理")
    
    # 处理每个批次
    for i, batch in enumerate(batches):
        logging.info(f"开始处理批次 {i+1}/{len(batches)}，包含 {len(batch)} 个样本")
        
        results = process_batch(batch, i)
        
        # 更新已处理的样本
        for result in results:
            sample_id, status, _ = result
            if status == "success":
                processed_samples.add(sample_id)
        
        # 保存进度
        save_progress(list(processed_samples))
        
        # 短暂休息，避免API调用过于频繁
        time.sleep(1)
    
    # 合并所有批次结果
    merge_all_batch_results()
    
    logging.info("COCO图生文任务全部完成！")


def merge_all_batch_results():
    """合并所有批次结果"""
    all_results = []
    for fname in os.listdir(Config.output_dir):
        if fname.startswith("coco_img2text_batch_") and fname.endswith("_results.json"):
            with open(os.path.join(Config.output_dir, fname), 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
    
    merged_path = os.path.join(Config.output_dir, "coco_img2text_all_results.json")
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"✅ 所有批次结果已合并保存为：{merged_path}")


# --------------------------
# 5. 执行
# --------------------------
if __name__ == "__main__":
    init_logger()
    
    # 检查是否需要从头开始处理
    if os.path.exists(Config.processed_samples_file):
        print("\n检测到之前的处理进度，请选择：")
        print("1. 从断点继续处理")
        print("2. 重新开始处理（将覆盖之前的进度）")
        choice = input("请输入选项（1或2）：")
        
        if choice == "2":
            # 删除进度文件，重新开始
            if os.path.exists(Config.processed_samples_file):
                os.remove(Config.processed_samples_file)
            logging.info("已清除之前的进度，将重新开始处理")
    
    # 开始处理
    process_coco_img2text_samples()
