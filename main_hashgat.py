import argparse
import torch
import random
import numpy as np
from utils import logger
from hashgat_trainer import HashGATTrainer


def param_list(log, config):
    """
    打印配置参数列表
    """
    log.info('>>> HashGAT Configs List <<<')
    log.info('--- Dataset: {}'.format(config.DATASET))
    log.info('--- SEED: {}'.format(config.SEED))
    log.info('--- Hash Bit: {}'.format(config.HASH_BIT))
    log.info('--- Batch Size: {}'.format(config.BATCH_SIZE))
    log.info('--- Learning Rate: {}'.format(config.LR_MyNet))
    log.info('--- Number of Epochs: {}'.format(config.NUM_EPOCH))
    
    log.info('--- gamma (reconstruction): {}'.format(config.gamma))
    log.info('--- eta (similarity): {}'.format(config.eta))
    log.info('--- beta (quantization): {}'.format(config.beta))
    log.info('--- K (top-k neighbors): {}'.format(config.K))
    log.info('--- a1: {}'.format(config.a1))
    log.info('--- a2: {}'.format(config.a2))
    log.info('--- lambda_: {}'.format(config.lambda_))
    log.info('--- alpha: {}'.format(config.alpha))


def main(config):
    """
    主函数
    """
    # 设置随机种子以确保结果可重复
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(config.GPU_ID)
    
    # 内存优化设置
    if torch.cuda.is_available():
        # 设置内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.9)  # 限制PyTorch使用90%的GPU内存
        torch.cuda.empty_cache()  # 清理缓存
        
        # 确保使用确定性算法，禁用cudnn benchmark
        # torch.backends.cudnn.benchmark = True  # 注释掉这行，确保结果可重复
    
    # 创建日志记录器
    logName = config.DATASET + '_HashGAT_' + str(config.HASH_BIT)
    log = logger(logName)
    param_list(log, config)
    
    # 打印内存优化信息
    log.info('--- Batch Size: {}'.format(config.BATCH_SIZE))
    log.info('--- Gradient Accumulation Steps: {}'.format(getattr(config, 'GRAD_ACCUMULATION_STEPS', 4)))
    log.info('--- Mixed Precision Training: {}'.format(getattr(config, 'ENABLE_MIXED_PRECISION', True)))
    log.info('--- Memory Threshold: {}'.format(getattr(config, 'MEMORY_THRESHOLD', 0.9)))
    
    # 创建HashGAT训练器
    trainer = HashGATTrainer(log, config)
    
    # 训练或测试
    if config.TRAIN:
        best_map_i2t = 0
        best_map_t2i = 0
        
        for epoch in range(config.NUM_EPOCH):
            # 训练一个epoch
            avg_loss = trainer.train(epoch)
            
            # 定期评估
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I = trainer.evaluate()
                log.info('Epoch [%d/%d], mAP@50 I->T: %.3f, mAP@50 T->I: %.3f'
                         % (epoch + 1, config.NUM_EPOCH, MAP_I2T, MAP_T2I))
                
                # 更新学习率
                trainer.update_learning_rate((MAP_I2T + MAP_T2I) / 2)
                
                # 保存最佳模型
                if (MAP_I2T + MAP_T2I) > (best_map_i2t + best_map_t2i):
                    best_map_i2t, best_map_t2i = MAP_I2T, MAP_T2I
                    log.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f'
                             % (best_map_i2t, best_map_t2i))
                    trainer.save_model()
                
                log.info('--------------------------------------------------------------------')
    else:
        # 测试模式
        ckp = config.DATASET + '_HashGAT_' + str(config.HASH_BIT) + 'bits'
        trainer.load_model(ckp)
        MAP_I2T, MAP_T2I = trainer.evaluate()
        log.info('Test Results - mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='HashGAT: Hash-Aware Graph Attention Network')
    
    # 基本参数
    parser.add_argument('--TRAIN', default=True, type=bool, help='train or test')
    parser.add_argument('--DATASET', default='COCO', type=str, 
                        help='Dataset name: MIRFlickr, NUSWIDE or COCO')
    parser.add_argument('--HASH_BIT', default=16, type=int, help='Hash code length')
    parser.add_argument('--BATCH_SIZE', default=512, type=int, help='Batch size')
    parser.add_argument('--GPU_ID', default=0, type=int, help='GPU ID')
    parser.add_argument('--SEED', default=42, type=int, help='Random seed')
    parser.add_argument('--NUM_WORKERS', default=4, type=int, help='Number of workers for data loading')
    parser.add_argument('--EPOCH_INTERVAL', default=2, type=int, help='Epoch interval for printing')
    parser.add_argument('--NUM_EPOCH', default=60, type=int, help='Number of epochs')
    parser.add_argument('--EVAL_INTERVAL', default=10, type=int, help='Evaluation interval')
    parser.add_argument('--MODEL_DIR', default="./checkpoints", type=str, help='Model directory')
    
    # HashGAT特定参数 - 优化后的参数
    parser.add_argument('--LR_MyNet', default=0.001, type=float, help='Learning rate for HashGAT')
    parser.add_argument('--gamma', default=10.0, type=float, help='Weight for reconstruction loss')
    parser.add_argument('--eta', default=1.0, type=float, help='Weight for similarity loss')
    parser.add_argument('--beta', default=0.15, type=float, help='Weight for quantization loss')
    parser.add_argument('--K', default=10, type=int, help='Number of top-k neighbors')
    parser.add_argument('--a1', default=0.1, type=float, help='Balance ST and SI')
    parser.add_argument('--a2', default=0.3, type=float, help='Balance S1 and S2')
    parser.add_argument('--lambda_', default=0.5, type=float, help='Balance S1 and S2')
    parser.add_argument('--alpha', default=0.5, type=float, help='Balance feature similarity and discriminator score')
    parser.add_argument('--USE_DISCRIMINATOR', default=True, type=bool, help='Use discriminator for similarity calculation')
    
    # 内存优化参数
    parser.add_argument('--GRAD_ACCUMULATION_STEPS', default=4, type=int, help='Gradient accumulation steps')
    parser.add_argument('--MEMORY_THRESHOLD', default=0.9, type=float, help='Memory usage threshold (0-1)')
    parser.add_argument('--ENABLE_MIXED_PRECISION', default=True, type=bool, help='Enable mixed precision training')
    
    # 消融实验参数
    parser.add_argument('--DISABLE_HASH_ATTENTION', default=False, type=bool, help='Disable hash-aware attention mechanism')
    parser.add_argument('--DISABLE_DYNAMIC_GRAPH', default=False, type=bool, help='Disable dynamic graph construction')

    # Hash-conditioned edge reweighting (HEW)
    parser.add_argument('--USE_HASH_EDGE_WEIGHT', default=True, type=bool,
                        help='Use hash-conditioned edge reweighting in GAT attention')
    parser.add_argument('--TAU_H', default=0.5, type=float,
                        help='Temperature for hash-based edge weights (smaller => sharper)')
    parser.add_argument('--EDGE_HASH_LEN', default=0, type=int,
                        help='Continuous hash length for edge weights (0 uses HASH_BIT)')
    
    # 解析参数
    config = parser.parse_args()
    
    # 运行主函数
    main(config)
