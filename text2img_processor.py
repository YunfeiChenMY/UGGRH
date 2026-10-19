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
    coco_text2img_images_dir = os.path.join(output_dir, "coco_text2img_images")
    coco_text2img_texts_dir = os.path.join(output_dir, "coco_text2img_texts")
    
    # 日志文件路径
    log_file = os.path.join(output_dir, "coco_text2img_processing_log.txt")
    
    # 进度保存和恢复相关配置
    progress_file = os.path.join(output_dir, "coco_text2img_progress.json")
    processed_samples_file = os.path.join(output_dir, "coco_text2img_processed_samples.json")
    
    # API配置
    text2img_api_url = ""
    api_key = ""
    
    # 处理配置
    max_workers = 2  # 线程数
    batch_size = 10  # 每批处理的样本数量
    
    # 文件名后缀，用于标识生成的数据
    generated_image_suffix = "_gen.jpg"


# --------------------------
# 2. 初始化日志
# --------------------------
def init_logger():
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.coco_text2img_images_dir, exist_ok=True)
    os.makedirs(Config.coco_text2img_texts_dir, exist_ok=True)
    
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
def get_coco_text2img_samples():
    """从数据集中获取需要进行文生图处理的样本"""
    text2img_samples = []
    
    # 从 coco_text_to_image 文件夹中获取文本
    texts_dir = os.path.join(Config.coco_text_to_image_dir, "texts")
    
    if not os.path.exists(texts_dir):
        logging.error(f"文生图文本文件夹不存在：{texts_dir}")
        return []
    
    # 获取所有文本文件
    text_files = []
    for f in os.listdir(texts_dir):
        if f.lower().endswith('.txt'):
            text_files.append(f)
    
    logging.info(f"在 {texts_dir} 中找到 {len(text_files)} 个文本文件")
    
    # 构建样本列表
    for text_file in text_files:
        # 从文件名提取ID（去掉扩展名）
        name, ext = os.path.splitext(text_file)
        file_id = name
        
        # 直接添加文本文件，不需要匹配现有图片
        text2img_samples.append({
            "id": file_id,
            "text_path": os.path.join(texts_dir, text_file)
        })
    
    logging.info(f"构建了 {len(text2img_samples)} 个COCO文生图样本")
    return text2img_samples


def test_api_connection():
    """测试API连接是否正常"""
    try:
        logging.info("正在测试API连接...")
        # 首先测试API是否可访问
        test_response = requests.get(f"{Config.text2img_api_url}/sdapi/v1/options", timeout=10)
        if test_response.status_code == 200:
            logging.info("API连接测试成功")
            return True
        else:
            logging.error(f"API连接测试失败，状态码：{test_response.status_code}")
            return False
    except Exception as e:
        logging.error(f"API连接测试异常：{str(e)}")
        return False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_text2img_api_with_retry(prompt_text):
    """调用文生图API，带重试机制"""
    try:
        data = {
            "prompt": prompt_text + ", photorealistic, high quality, bright lighting, realistic texture",
            "negative_prompt": "lowres, blurry, bad anatomy, disfigured, poorly drawn",
            "steps": 30,
            "cfg_scale": 9.0,
            "width": 512,
            "height": 512,
            "sampler_name": "Euler a",
            "seed": -1
        }

        logging.info(f"正在调用API，URL：{Config.text2img_api_url}/sdapi/v1/txt2img")
        logging.info(f"请求数据：{data}")
        
        response = requests.post(
            f"{Config.text2img_api_url}/sdapi/v1/txt2img",
            json=data,
            timeout=60
        )
        
        logging.info(f"API响应状态码：{response.status_code}")
        
        if response.status_code != 200:
            logging.error(f"API请求失败，状态码：{response.status_code}")
            logging.error(f"响应内容：{response.text}")
            return None
            
        response.raise_for_status()

        result = response.json()
        logging.info(f"API响应JSON：{result}")
        
        if "images" in result and result["images"]:
            img_base64 = result["images"][0]
            return base64.b64decode(img_base64)
        else:
            logging.warning(f"生成图像失败（无images字段）：{prompt_text[:20]}...")
            logging.warning(f"API响应内容：{result}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"API请求异常：{str(e)}")
        raise
    except Exception as e:
        logging.error(f"文生图处理异常：{str(e)}")
        raise


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
    text_path = sample["text_path"]
    
    try:
        # 复制原始文本到输出目录
        output_text_path = os.path.join(Config.coco_text2img_texts_dir, f"{sample_id}.txt")
        shutil.copy2(text_path, output_text_path)
        
        # 读取原始文本
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 调用文生图API
        image_data = call_text2img_api_with_retry(text)
        
        if image_data:
            # 保存生成的图片
            image_filename = f"{sample_id}.jpg"
            with open(os.path.join(Config.coco_text2img_images_dir, image_filename), 'wb') as f:
                f.write(image_data)
            
            logging.info(f"样本 {sample_id} 文生图处理成功")
            return sample_id, "success", "text2img"
        else:
            # 如果生成失败，记录失败但不复制原始图片（因为没有原始图片）
            logging.warning(f"样本 {sample_id} 文生图处理失败")
            return sample_id, "failed", "text2img_failed"
    
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
    batch_log = os.path.join(Config.output_dir, f"coco_text2img_batch_{batch_idx}_results.json")
    with open(batch_log, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"批次 {batch_idx} 处理完成，结果保存至 {batch_log}")
    return results


# --------------------------
# 4. 主流程
# --------------------------
def process_coco_text2img_samples():
    """处理COCO文生图样本的主流程"""
    init_logger()
    
    # 测试API连接
    if not test_api_connection():
        logging.error("API连接测试失败，请检查API地址和网络连接")
        return
    
    # 获取需要进行文生图处理的样本
    text2img_samples = get_coco_text2img_samples()
    
    if len(text2img_samples) == 0:
        logging.error("没有找到COCO文生图样本，请检查文件夹是否存在")
        return
    
    # 加载已处理的样本
    processed_samples = load_progress()
    
    # 过滤出未处理的样本
    unprocessed_samples = [s for s in text2img_samples if s["id"] not in processed_samples]
    
    logging.info(f"总共需要处理 {len(text2img_samples)} 个样本，已处理 {len(processed_samples)} 个，待处理 {len(unprocessed_samples)} 个")
    
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
    
    logging.info("COCO文生图任务全部完成！")


def merge_all_batch_results():
    """合并所有批次结果"""
    all_results = []
    for fname in os.listdir(Config.output_dir):
        if fname.startswith("coco_text2img_batch_") and fname.endswith("_results.json"):
            with open(os.path.join(Config.output_dir, fname), 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
    
    merged_path = os.path.join(Config.output_dir, "coco_text2img_all_results.json")
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
    process_coco_text2img_samples()
