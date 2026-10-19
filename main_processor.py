import os
import json
import shutil
import logging
from datetime import datetime


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
    coco_text2img_images_dir = os.path.join(output_dir, "coco_text2img_images")
    coco_text2img_texts_dir = os.path.join(output_dir, "coco_text2img_texts")
    
    # 最终输出目录
    final_output_dir = os.path.join(coco_root, "coco_new_pairs")
    final_images_dir = os.path.join(final_output_dir, "images")
    final_texts_dir = os.path.join(final_output_dir, "texts")
    
    # 日志文件路径
    log_file = os.path.join(final_output_dir, "coco_main_processing_log.txt")
    
    # 处理脚本路径
    img2text_script = os.path.join(os.path.dirname(__file__), "coco_img2text_processor.py")
    text2img_script = os.path.join(os.path.dirname(__file__), "coco_text2img_processor.py")


# --------------------------
# 2. 初始化日志
# --------------------------
def init_logger():
    os.makedirs(Config.final_output_dir, exist_ok=True)
    os.makedirs(Config.final_images_dir, exist_ok=True)
    os.makedirs(Config.final_texts_dir, exist_ok=True)
    
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
def run_script(script_path):
    """运行指定的Python脚本"""
    import subprocess
    try:
        logging.info(f"开始运行脚本: {script_path}")
        result = subprocess.run(["python", script_path], check=True, capture_output=True, text=True)
        logging.info(f"脚本 {script_path} 运行成功")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"脚本 {script_path} 运行失败: {e}")
        logging.error(f"错误输出: {e.stderr}")
        return False


def merge_generated_pairs():
    """合并生成的图文对到最终输出目录"""
    logging.info("开始合并生成的图文对...")
    
    # 1. 合并图生文结果
    img2text_images = os.listdir(Config.coco_img2text_images_dir)
    img2text_texts = os.listdir(Config.coco_img2text_texts_dir)
    
    img2text_count = 0
    for img_file in img2text_images:
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            # 提取文件ID
            file_id, ext = os.path.splitext(img_file)
            
            # 检查对应的文本文件是否存在
            text_file = f"{file_id}.txt"
            if text_file in img2text_texts:
                # 复制图片和文本到最终目录
                shutil.copy2(
                    os.path.join(Config.coco_img2text_images_dir, img_file),
                    os.path.join(Config.final_images_dir, f"img2text_{file_id}{ext}")
                )
                shutil.copy2(
                    os.path.join(Config.coco_img2text_texts_dir, text_file),
                    os.path.join(Config.final_texts_dir, f"img2text_{file_id}.txt")
                )
                img2text_count += 1
    
    logging.info(f"成功合并 {img2text_count} 个图生文图文对")
    
    # 2. 合并文生图结果
    text2img_images = os.listdir(Config.coco_text2img_images_dir)
    text2img_texts = os.listdir(Config.coco_text2img_texts_dir)
    
    text2img_count = 0
    for img_file in text2img_images:
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            # 提取文件ID
            file_id, ext = os.path.splitext(img_file)
            
            # 检查对应的文本文件是否存在
            text_file = f"{file_id}.txt"
            if text_file in text2img_texts:
                # 复制图片和文本到最终目录
                shutil.copy2(
                    os.path.join(Config.coco_text2img_images_dir, img_file),
                    os.path.join(Config.final_images_dir, f"text2img_{file_id}{ext}")
                )
                shutil.copy2(
                    os.path.join(Config.coco_text2img_texts_dir, text_file),
                    os.path.join(Config.final_texts_dir, f"text2img_{file_id}.txt")
                )
                text2img_count += 1
    
    logging.info(f"成功合并 {text2img_count} 个文生图图文对")
    
    # 3. 生成最终结果清单
    final_pairs = []
    
    # 添加图生文对
    for img_file in os.listdir(Config.final_images_dir):
        if img_file.startswith("img2text_") and img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_id = img_file[9:]  # 去掉 "img2text_" 前缀
            file_id = os.path.splitext(file_id)[0]  # 去掉扩展名
            
            text_file = f"img2text_{file_id}.txt"
            if text_file in os.listdir(Config.final_texts_dir):
                final_pairs.append({
                    "id": f"img2text_{file_id}",
                    "type": "img2text",
                    "image_file": img_file,
                    "text_file": text_file
                })
    
    # 添加文生图对
    for img_file in os.listdir(Config.final_images_dir):
        if img_file.startswith("text2img_") and img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            file_id = img_file[10:]  # 去掉 "text2img_" 前缀
            file_id = os.path.splitext(file_id)[0]  # 去掉扩展名
            
            text_file = f"text2img_{file_id}.txt"
            if text_file in os.listdir(Config.final_texts_dir):
                final_pairs.append({
                    "id": f"text2img_{file_id}",
                    "type": "text2img",
                    "image_file": img_file,
                    "text_file": text_file
                })
    
    # 保存最终结果清单
    pairs_list_path = os.path.join(Config.final_output_dir, "coco_new_pairs_list.json")
    with open(pairs_list_path, 'w', encoding='utf-8') as f:
        json.dump(final_pairs, f, ensure_ascii=False, indent=2)
    
    logging.info(f"最终合并完成，共生成 {len(final_pairs)} 个新图文对")
    logging.info(f"结果清单保存至: {pairs_list_path}")
    
    return len(final_pairs)


# --------------------------
# 4. 主流程
# --------------------------
def main():
    """主处理流程"""
    init_logger()
    
    logging.info("=" * 50)
    logging.info("开始COCO数据集图文对生成任务")
    logging.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 50)
    
    # 1. 运行图生文处理脚本
    logging.info("步骤1: 开始图生文处理...")
    if not run_script(Config.img2text_script):
        logging.error("图生文处理失败，终止任务")
        return False
    
    # 2. 运行文生图处理脚本
    logging.info("步骤2: 开始文生图处理...")
    if not run_script(Config.text2img_script):
        logging.error("文生图处理失败，终止任务")
        return False
    
    # 3. 合并生成的图文对
    logging.info("步骤3: 合并生成的图文对...")
    total_pairs = merge_generated_pairs()
    
    # 4. 任务完成
    logging.info("=" * 50)
    logging.info("COCO数据集图文对生成任务完成")
    logging.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"共生成 {total_pairs} 个新图文对")
    logging.info(f"结果保存在: {Config.final_output_dir}")
    logging.info("=" * 50)
    
    return True


# --------------------------
# 5. 执行
# --------------------------
if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ COCO数据集图文对生成任务成功完成！")
        print(f"结果保存在: {Config.final_output_dir}")
    else:
        print("\n❌ COCO数据集图文对生成任务失败，请检查日志获取详细信息。")
