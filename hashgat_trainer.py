import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from load_data import get_loader_flickr, get_loader_nus, get_loader_coco
from hashgat_model import HashGAT, HashGATLoss
from utils import compress, calculate_top_map, logger
import numpy as np
import os.path as osp
import os
from discriminator import Discriminator
import threading
import queue

class DataPrefetcher:
    """
    数据预取器，减少GPU等待数据的时间
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        
    def preload(self):
        try:
            self.next_img, self.next_txt, self.next_label, self.next_index = next(self.loader)
        except StopIteration:
            self.next_img = None
            self.next_txt = None
            self.next_label = None
            self.next_index = None
            return
            
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.cuda(non_blocking=True)
            self.next_txt = self.next_txt.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
            self.next_index = self.next_index.cuda(non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img = self.next_img
        txt = self.next_txt
        label = self.next_label
        index = self.next_index
        if img is not None:
            img.record_stream(torch.cuda.current_stream())
            txt.record_stream(torch.cuda.current_stream())
            label.record_stream(torch.cuda.current_stream())
            index.record_stream(torch.cuda.current_stream())
        self.preload()
        return img, txt, label, index


class HashGATTrainer:
    """
    HashGAT训练器
    实现论文中的算法1
    """
    def __init__(self, log, config):
        self.config = config
        self.log = log
        
        # 加载数据
        if config.DATASET == 'MIRFlickr':
            dataloader, data_train = get_loader_flickr(config.BATCH_SIZE)
        elif config.DATASET == 'NUSWIDE':
            dataloader, data_train = get_loader_nus(config.BATCH_SIZE)
        elif config.DATASET == 'COCO':
            dataloader, data_train = get_loader_coco(config.BATCH_SIZE)
        
        self.I_tr, self.T_tr, self.L_tr = data_train
        self.train_num = self.I_tr.shape[0]
        self.img_feat_len = self.I_tr.shape[1]
        self.txt_feat_len = self.T_tr.shape[1]
        
        # 初始化数据加载器
        self.train_loader = dataloader['train']
        self.test_loader = dataloader['query']
        self.database_loader = dataloader['database']
        
        # 计算跨模态相似度矩阵
        img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        #计算Scmm
        #如果 Scmm(i,j) 大，
        # loss 会逼迫 H_i 和 H_j（以及 HI/HT）更相似；
        # 如果 S_cmm(i,j) 小或为 0（top-K 之后很多是 0），
        # 就不会鼓励它们靠近，从而把哈希空间的结构“按 S_{\text{cmm}} 排列好
        self.S_cmm = self.cal_similarity(img_norm, txt_norm)
        
        # 初始化HashGAT模型
        # 从配置中获取消融实验参数
        use_hash_attention = not getattr(config, 'DISABLE_HASH_ATTENTION', False)
        use_dynamic_graph = not getattr(config, 'DISABLE_DYNAMIC_GRAPH', False)
        use_hash_edge_weight = getattr(config, 'USE_HASH_EDGE_WEIGHT', True)
        tau_h = getattr(config, 'TAU_H', 0.5)
        edge_hash_len = getattr(config, 'EDGE_HASH_LEN', 0)
        edge_hash_len = None if not edge_hash_len else edge_hash_len
        
        # 设置随机种子以确保模型初始化的一致性
        seed = getattr(config, 'SEED', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        self.hashgat = HashGAT(
            img_feat_dim=self.img_feat_len,
            txt_feat_dim=self.txt_feat_len,
            hash_len=config.HASH_BIT,
            embed_dim=512,
            num_layers=2,
            num_heads=8,
            lambda_hash=0.5,
            top_k=10,  # 使用固定的K值
            use_hash_attention=use_hash_attention,
            use_dynamic_graph=use_dynamic_graph,
            use_hash_edge_weight=use_hash_edge_weight,
            tau_h=tau_h,
            edge_hash_len=edge_hash_len
        ).cuda()
        
        # 初始化损失函数
        self.criterion = HashGATLoss(
            feature_dim=512,  # 与HashGAT中的embed_dim一致
            hash_len=config.HASH_BIT,
            gamma=config.gamma,
            eta=config.eta,
            beta=config.beta
        ).cuda()
        
        # 初始化优化器 - 使用AdamW，通常比Adam收敛更快
        self.optimizer = torch.optim.AdamW(
            self.hashgat.parameters(),
            lr=config.LR_MyNet,
            weight_decay=0.01
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 记录训练损失
        self.record_loss = []
        
        # 梯度累积参数
        self.accumulation_steps = getattr(config, 'GRAD_ACCUMULATION_STEPS', 4)  # 累积4个批次再更新一次
        
        # 混合精度训练参数
        if getattr(config, 'ENABLE_MIXED_PRECISION', True):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 内存监控参数
        self.memory_threshold = getattr(config, 'MEMORY_THRESHOLD', 0.9)  # 当内存使用超过90%时触发清理
        
        # 加载判别器（如果需要）
        if hasattr(config, 'USE_DISCRIMINATOR') and config.USE_DISCRIMINATOR:
            feature_dim = self.img_feat_len
            self.discriminator = Discriminator(feature_dim).cuda()
            self.discriminator.load_state_dict(torch.load('discriminator.pth'))
            self.discriminator.eval()
    
    def cal_similarity(self, F_I, F_T):
        """
        计算跨模态相似度矩阵 - 优化版本，减少计算复杂度
        结合特征相似度和判别器分数
        """
        batch_size = F_I.size(0)
        top_k = self.config.K
        lambda_ = getattr(self.config, 'lambda_', 0.5)
        alpha = getattr(self.config, 'alpha', 0.5)
        device = F_I.device
        
        # 归一化特征
        F_I = F.normalize(F_I, dim=1)
        F_T = F.normalize(F_T, dim=1)
        
        # Map cosine similarities to the nonnegative affinity scale.
        S_I = F_I @ F_I.t()
        S_T = F_T @ F_T.t()
        S_I = (S_I + 1.0) / 2.0
        S_T = (S_T + 1.0) / 2.0
        S1 = self.config.a1 * S_I + (1 - self.config.a1) * S_T
        
        # 如果使用判别器，计算判别器分数 - 优化版本
        if hasattr(self, 'discriminator'):
            feature_dim = F_I.shape[1]
            # 增大chunk_size以减少循环次数
            chunk_size = min(256, batch_size)
            D = torch.zeros(batch_size, batch_size, device=device)
            
            for start_i in range(0, batch_size, chunk_size):
                end_i = min(start_i + chunk_size, batch_size)
                F_I_chunk = F_I[start_i:end_i]
                
                # 使用更高效的张量操作
                F_I_exp = F_I_chunk.unsqueeze(1).expand(-1, batch_size, -1)
                F_T_exp = F_T.unsqueeze(0).expand(end_i - start_i, -1, -1)
                
                with torch.no_grad():
                    scores = self.discriminator(F_I_exp.reshape(-1, feature_dim),
                                             F_T_exp.reshape(-1, feature_dim))
                scores = scores.reshape(end_i - start_i, batch_size)
                
                D[start_i:end_i, :] = scores
            
            # Fuse the intra-modal affinity with the cross-modal score.
            S1 = alpha * S1 + (1 - alpha) * D
        
        # 简化相似度处理 - 减少复杂操作
        if top_k > 0:
            # 只对每行保留top-k最大的相似度
            _, top_indices = torch.topk(S1, k=min(top_k, batch_size), dim=1)
            S1_masked = torch.zeros_like(S1)
            S1_masked.scatter_(1, top_indices, S1.gather(1, top_indices))
            S1 = S1_masked
        
        # 简化激活函数和相似度融合
        S1_activated = torch.tanh(S1)
        S2 = 2.0 * S1_activated + torch.eye(batch_size, device=device)
        S2 = lambda_ * S2 + (1 - lambda_) * S1
        
        S = self.config.a2 * S1 + (1 - self.config.a2) * S2
        S = 0.5 * (S + S.t())
        
        return S
    
    def check_memory_usage(self):
        """
        检查GPU内存使用情况，并在需要时清理缓存
        """
        if torch.cuda.is_available():
            # 获取当前GPU内存使用情况
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            
            # 计算内存使用比例
            memory_ratio = memory_reserved / total_memory
            
            # 如果内存使用超过阈值，清理缓存
            if memory_ratio > self.memory_threshold:
                self.log.info(f"Memory usage {memory_ratio:.2f} exceeds threshold {self.memory_threshold}, clearing cache...")
                torch.cuda.empty_cache()
                return True
        return False
    
    def train(self, epoch):
        """
        训练HashGAT模型 - 高度优化版本，大幅减少训练时间
        实现论文中的算法1
        """
        self.hashgat.train()
        total_loss = 0
        
        # 在每个epoch开始时清理GPU内存，减少内存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 设置随机种子以确保每个epoch的训练过程一致
        # 使用固定种子，不随epoch变化
        seed = getattr(self.config, 'SEED', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # 减少评估频率，只在特定epoch进行评估
        should_evaluate = (epoch + 1) % self.config.EVAL_INTERVAL == 0
        
        # 预取数据以提高训练速度
        prefetcher = DataPrefetcher(self.train_loader)
        
        try:
            img, txt, _, index_ = prefetcher.next()
        except StopIteration:
            # 如果没有数据，直接返回
            return 0
        
        batch_idx = 0
        while img is not None:
            # 减少内存检查频率
            if batch_idx % 50 == 0:
                self.check_memory_usage()
            
            # 准备数据
            img = Variable(img.cuda().float())
            txt = Variable(txt.cuda().float())
            
            # 获取批次相似度矩阵 - 使用预计算的矩阵，避免重复计算
            S_batch = self.S_cmm[index_.unsqueeze(1), index_.unsqueeze(0)]
            
            # 根据配置选择是否使用混合精度训练
            if self.scaler is not None:
                # 使用自动混合精度提高训练速度并减少内存使用
                with torch.cuda.amp.autocast():
                    # 前向传播
                    HI, HT, H, B, DeI_feat, DeT_feat, img_specific, txt_specific = self.hashgat(img, txt)
                    
                    # 计算损失
                    loss, loss_dict = self.criterion(HI, HT, H, S_batch, DeI_feat, DeT_feat, img, txt)
                    
                    # 梯度累积：损失除以累积步数
                    loss = loss / self.accumulation_steps
                    
                # 使用混合精度进行反向传播
                self.scaler.scale(loss).backward()
            else:
                # 不使用混合精度训练
                # 前向传播
                HI, HT, H, B, DeI_feat, DeT_feat, img_specific, txt_specific = self.hashgat(img, txt)
                
                # 计算损失
                loss, loss_dict = self.criterion(HI, HT, H, S_batch, DeI_feat, DeT_feat, img, txt)
                
                # 梯度累积：损失除以累积步数
                loss = loss / self.accumulation_steps
                
                # 反向传播
                loss.backward()
            
            # 记录损失（乘以累积步数以获得实际损失值）
            total_loss += loss.item() * self.accumulation_steps
            
            # 梯度累积：每accumulation_steps个批次更新一次参数
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪，防止梯度爆炸
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.hashgat.parameters(), max_norm=1.0)
                
                # 更新参数
                if self.scaler is not None:
                    # 使用混合精度更新参数
                    self.scaler.step(self.optimizer)
                    
                    # 更新缩放器
                    self.scaler.update()
                else:
                    # 不使用混合精度更新参数
                    self.optimizer.step()
                
                # 清零梯度
                self.optimizer.zero_grad()
            
            # 减少打印频率
            if (batch_idx + 1) % (self.train_num // self.config.BATCH_SIZE / max(1, self.config.EPOCH_INTERVAL * 2)) == 0:
                self.log.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, CCR: %.4f, ACR: %.4f, Quant: %.4f'
                             % (epoch + 1, self.config.NUM_EPOCH, batch_idx + 1,
                                self.train_num // self.config.BATCH_SIZE,
                                loss_dict['total_loss'], loss_dict['loss_rec'],
                                loss_dict['loss_sim'], loss_dict['quant_loss']))
            
            # 获取下一批数据
            try:
                img, txt, _, index_ = prefetcher.next()
                batch_idx += 1
            except StopIteration:
                break
        
        # 处理最后一个不完整的累积批次
        if batch_idx % self.accumulation_steps != 0:
            # 梯度裁剪，防止梯度爆炸
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.hashgat.parameters(), max_norm=1.0)
            
            # 更新参数
            if self.scaler is not None:
                # 使用混合精度更新参数
                self.scaler.step(self.optimizer)
                
                # 更新缩放器
                self.scaler.update()
            else:
                # 不使用混合精度更新参数
                self.optimizer.step()
            
            # 清零梯度
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / (batch_idx + 1)  # 使用实际处理的批次数
        self.record_loss.append(avg_loss)
        
        return avg_loss
    
    def update_learning_rate(self, current_map):
        """
        根据当前mAP更新学习率
        """
        self.scheduler.step(current_map)
    
    def evaluate(self):
        """
        评估HashGAT模型 - 确保使用当前模型生成的哈希码
        """
        self.hashgat.eval()
        
        # 检查内存使用情况
        self.check_memory_usage()
        
        # 设置随机种子以确保评估过程一致
        seed = getattr(self.config, 'SEED', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # 设置PyTorch为确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 提取数据库和查询集的哈希码 - 强制刷新缓存，使用当前模型生成的哈希码
        db_BI, db_BT, db_L, qu_BI, qu_BT, qu_L, db_img_specific, db_txt_specific, qu_img_specific, qu_txt_specific = self.extract_hash_codes(force_refresh=True)
        
        # 确保数据类型正确，避免不必要的转换
        db_BI_np = np.asarray(db_BI, dtype=np.float32)
        db_BT_np = np.asarray(db_BT, dtype=np.float32)
        db_L_np = np.asarray(db_L, dtype=np.float32)
        qu_BI_np = np.asarray(qu_BI, dtype=np.float32)
        qu_BT_np = np.asarray(qu_BT, dtype=np.float32)
        qu_L_np = np.asarray(qu_L, dtype=np.float32)
        
        # 使用模态特定的哈希码
        db_img_specific_np = np.asarray(db_img_specific, dtype=np.float32)
        db_txt_specific_np = np.asarray(db_txt_specific, dtype=np.float32)
        qu_img_specific_np = np.asarray(qu_img_specific, dtype=np.float32)
        qu_txt_specific_np = np.asarray(qu_txt_specific, dtype=np.float32)
        
        # 计算mAP - 使用当前模型生成的哈希码
        import time
        start_time = time.time()
        
        # 使用模态特定的哈希码计算mAP，使用完全对称的权重分配
        # 使用50%/50%的对称权重，确保公平评估
        # I2T: 使用对称的权重组合
        MAP_I2T = calculate_top_map(qu_B=(0.5 * qu_BI_np + 0.5 * qu_img_specific_np),
                                  re_B=(0.5 * db_BT_np + 0.5 * db_txt_specific_np),
                                  qu_L=qu_L_np, re_L=db_L_np, topk=50)
        # T2I: 使用对称的权重组合
        MAP_T2I = calculate_top_map(qu_B=(0.5 * qu_BT_np + 0.5 * qu_txt_specific_np),
                                  re_B=(0.5 * db_BI_np + 0.5 * db_img_specific_np),
                                  qu_L=qu_L_np, re_L=db_L_np, topk=50)
        
        end_time = time.time()
        self.log.info(f"mAP calculation time: {end_time - start_time:.2f} seconds")
        
        return MAP_I2T, MAP_T2I
    
    def extract_hash_codes(self, force_refresh=False):
        """
        提取数据库和查询集的哈希码
        高度优化版本：使用更大的批次、并行处理和缓存机制
        
        参数:
            force_refresh: 是否强制刷新缓存，重新计算哈希码
        """
        # 设置随机种子以确保哈希码提取过程一致
        seed = getattr(self.config, 'SEED', 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        
        # 设置PyTorch为确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 检查是否已经缓存了哈希码
        cache_file = f"./cache/{self.config.DATASET}_hashcodes_{self.config.HASH_BIT}bit_seed{seed}.npz"
        if os.path.exists(cache_file) and not force_refresh:
            self.log.info("Loading cached hash codes...")
            data = np.load(cache_file)
            # 检查缓存文件是否包含模态特定哈希码
            if 'db_img_specific' in data:
                return (data['db_BI'], data['db_BT'], data['db_L'], data['qu_BI'], data['qu_BT'], data['qu_L'],
                        data['db_img_specific'], data['db_txt_specific'], data['qu_img_specific'], data['qu_txt_specific'])
            else:
                # 如果缓存文件不包含模态特定哈希码，则强制刷新
                self.log.info("Cache file does not contain modality-specific hash codes, forcing refresh...")
                force_refresh = True
        
        # 创建缓存目录
        os.makedirs("./cache", exist_ok=True)
        
        # 提取数据库哈希码 - 使用更大的批次
        db_BI = []
        db_BT = []
        db_L = []
        db_img_specific = []  # 新增：图像特定哈希码
        db_txt_specific = []  # 新增：文本特定哈希码
        
        # 使用更大的批次大小以减少前向传播次数
        original_batch_size = self.database_loader.batch_size
        optimized_batch_size = min(original_batch_size * 2, 256)  # 增加批次大小但不超过256
        
        # 设置随机种子以确保数据加载一致
        seed = getattr(self.config, 'SEED', 42)
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            import random
            random.seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # 重新创建数据加载器以使用更大的批次
        from torch.utils.data import DataLoader
        db_dataset = self.database_loader.dataset
        optimized_db_loader = DataLoader(
            db_dataset,
            batch_size=optimized_batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True
        )
        
        with torch.no_grad():
            for batch_idx, (img, txt, label, _) in enumerate(optimized_db_loader):
                # 检查内存使用情况
                if batch_idx % 5 == 0:  # 减少检查频率
                    self.check_memory_usage()
                
                img = img.cuda().float()
                txt = txt.cuda().float()
                
                # 使用自动混合精度加速计算
                with torch.cuda.amp.autocast():
                    _, _, _, B, _, _, img_specific, txt_specific = self.hashgat(img, txt)
                
                # 直接使用numpy数组，避免频繁的list扩展
                db_BI.append(B.cpu().numpy())
                db_BT.append(B.cpu().numpy())
                db_L.append(label.cpu().numpy())
                db_img_specific.append(img_specific.cpu().numpy())  # 新增
                db_txt_specific.append(txt_specific.cpu().numpy())  # 新增
                
                # 及时清理内存
                del img, txt, B, img_specific, txt_specific
                if batch_idx % 10 == 0:  # 减少清理频率
                    torch.cuda.empty_cache()
        
        # 提取查询集哈希码 - 同样使用更大的批次
        qu_BI = []
        qu_BT = []
        qu_L = []
        qu_img_specific = []  # 新增：图像特定哈希码
        qu_txt_specific = []  # 新增：文本特定哈希码
        
        qu_dataset = self.test_loader.dataset
        optimized_qu_loader = DataLoader(
            qu_dataset,
            batch_size=optimized_batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True
        )
        
        with torch.no_grad():
            for batch_idx, (img, txt, label, _) in enumerate(optimized_qu_loader):
                # 检查内存使用情况
                if batch_idx % 5 == 0:  # 减少检查频率
                    self.check_memory_usage()
                
                img = img.cuda().float()
                txt = txt.cuda().float()
                
                # 使用自动混合精度加速计算
                with torch.cuda.amp.autocast():
                    _, _, _, B, _, _, img_specific, txt_specific = self.hashgat(img, txt)
                
                # 直接使用numpy数组，避免频繁的list扩展
                qu_BI.append(B.cpu().numpy())
                qu_BT.append(B.cpu().numpy())
                qu_L.append(label.cpu().numpy())
                qu_img_specific.append(img_specific.cpu().numpy())  # 新增
                qu_txt_specific.append(txt_specific.cpu().numpy())  # 新增
                
                # 及时清理内存
                del img, txt, B, img_specific, txt_specific
                if batch_idx % 10 == 0:  # 减少清理频率
                    torch.cuda.empty_cache()
        
        # 使用np.concatenate一次性合并所有批次，比list.extend更高效
        db_BI = np.concatenate(db_BI, axis=0)
        db_BT = np.concatenate(db_BT, axis=0)
        db_L = np.concatenate(db_L, axis=0)
        qu_BI = np.concatenate(qu_BI, axis=0)
        qu_BT = np.concatenate(qu_BT, axis=0)
        qu_L = np.concatenate(qu_L, axis=0)
        
        # 合并模态特定哈希码
        db_img_specific = np.concatenate(db_img_specific, axis=0)
        db_txt_specific = np.concatenate(db_txt_specific, axis=0)
        qu_img_specific = np.concatenate(qu_img_specific, axis=0)
        qu_txt_specific = np.concatenate(qu_txt_specific, axis=0)
        
        # 缓存哈希码以备将来使用
        self.log.info("Caching hash codes...")
        np.savez_compressed(cache_file,
                           db_BI=db_BI, db_BT=db_BT, db_L=db_L,
                           qu_BI=qu_BI, qu_BT=qu_BT, qu_L=qu_L,
                           db_img_specific=db_img_specific, db_txt_specific=db_txt_specific,
                           qu_img_specific=qu_img_specific, qu_txt_specific=qu_txt_specific)
        
        return db_BI, db_BT, db_L, qu_BI, qu_BT, qu_L, db_img_specific, db_txt_specific, qu_img_specific, qu_txt_specific
    
    def save_model(self):
        """
        保存模型
        """
        file_name = self.config.DATASET + '_HashGAT_' + str(self.config.HASH_BIT) + 'bits.pth'
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        
        torch.save({
            'model_state_dict': self.hashgat.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'record_loss': self.record_loss
        }, ckp_path)
        
        self.log.info('**********Save HashGAT model successfully.**********')
    
    def load_model(self, file_name):
        """
        加载模型
        """
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        
        try:
            checkpoint = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.hashgat.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.record_loss = checkpoint['record_loss']
            self.log.info('**************** Load HashGAT model %s ****************' % ckp_path)
        except IOError:
            self.log.error('********** Fail to load HashGAT model %s!*********' % ckp_path)
            raise IOError
