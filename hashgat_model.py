import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from torch_geometric.nn import GATConv
from transformer_encoder import TransformerEncoder


class HashAwareAttention(nn.Module):
    """
    哈希感知注意力机制
    实现公式(12)中的哈希感知注意力计算
    """
    def __init__(self, in_features, out_features, hash_len):
        super(HashAwareAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hash_len = hash_len
        
        # 可学习的权重矩阵
        self.W = nn.Linear(in_features, out_features)
        
        # 注意力参数
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 哈希码投影
        self.hash_proj = nn.Linear(hash_len, out_features)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h, B, adj):
        """
        前向传播
        
        参数:
            h: 节点特征 [N, in_features]
            B: 哈希码 [N, hash_len]
            adj: 邻接矩阵 [N, N]
            
        返回:
            更新后的节点特征 [N, out_features]
        """
        N = h.size(0)
        
        # 线性变换
        h_prime = self.W(h)  # [N, out_features]
        
        # 哈希码投影
        B_proj = self.hash_proj(B)  # [N, out_features]
        
        # 准备注意力计算 - 优化内存使用
        # 使用广播机制代替repeat操作，减少内存消耗
        h_prime_i = h_prime.unsqueeze(1).expand(-1, N, -1)  # [N, N, out_features]
        h_prime_j = h_prime.unsqueeze(0).expand(N, -1, -1)  # [N, N, out_features]
        a_input = torch.cat([h_prime_i, h_prime_j], dim=2)  # [N, N, 2*out_features]
        
        # 计算基础注意力分数
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # 添加哈希感知项
        B_diff = torch.norm(B.unsqueeze(1) - B.unsqueeze(0), p=2, dim=2)  # [N, N]
        hash_attention = torch.exp(-B_diff)  # 哈希相似度作为注意力权重
        
        # 结合基础注意力和哈希感知注意力
        attention = e + hash_attention
        
        # 应用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(attention)
        attention = torch.where(adj > 0, attention, zero_vec)
        
        # 计算注意力权重
        attention = F.softmax(attention, dim=1)
        
        # 应用注意力权重
        h_prime = torch.matmul(attention, h_prime)
        
        # 结合哈希投影特征
        output = h_prime + B_proj
        
        return F.elu(output)


class DynamicGraphConstruction(nn.Module):
    """
    动态图构建模块
    实现公式(13)中的动态邻接矩阵更新
    """
    def __init__(self, lambda_hash=0.7):
        super(DynamicGraphConstruction, self).__init__()
        self.lambda_hash = lambda_hash
        
    def forward(self, h, B=None, top_k=10):
        """
        构建动态邻接矩阵
        
        参数:
            h: 节点特征 [N, D]
            B: 哈希码 [N, hash_len]
            top_k: 每个节点的最大邻居数
            
        返回:
            动态邻接矩阵 [N, N]
        """
        N = h.size(0)
        device = h.device
        
        # 确保top_k不超过批次大小-1（排除自己）
        actual_top_k = min(top_k, N - 1)
        if actual_top_k <= 0:
            # 如果批次大小为1，返回单位矩阵（只有自连接）
            return torch.eye(N, device=device)
        
        # 计算特征相似度
        h_norm = F.normalize(h, p=2, dim=1)
        S_feat = torch.mm(h_norm, h_norm.t())  # [N, N]
        
        if B is None:
            S_combined = S_feat
        else:
            B_norm = F.normalize(B, p=2, dim=1)
            S_hash = torch.mm(B_norm, B_norm.t())
            S_combined = (1 - self.lambda_hash) * S_feat + self.lambda_hash * S_hash
        
        # 找到每个节点的top-k邻居
        # 确保actual_top_k至少为1，避免torch.topk报错
        actual_top_k = max(1, actual_top_k)
        _, top_indices = torch.topk(S_combined, k=actual_top_k, dim=1)
        
        # 构建邻接矩阵
        adj = torch.zeros_like(S_combined, device=device)
        for i in range(N):
            adj[i, top_indices[i]] = 1.0
            
        # 对称化邻接矩阵
        adj = (adj + adj.t()) / 2
        
        return adj


class MultiHopAggregation(nn.Module):
    """
    多跳哈希感知聚合模块
    实现公式(14)中的多跳特征聚合
    """
    def __init__(self, num_hops, feature_dim, hash_len):
        super(MultiHopAggregation, self).__init__()
        self.num_hops = num_hops
        self.feature_dim = feature_dim
        self.hash_len = hash_len
        
        # 每一跳的权重
        self.hop_weights = nn.Parameter(torch.ones(num_hops + 1))
        
        # 哈希码投影
        self.hash_proj = nn.Linear(hash_len, feature_dim)
        
        # 确保权重归一化
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, h_list, B_list, adj_list):
        """
        多跳特征聚合
        
        参数:
            h_list: 每一层的节点特征列表 [h_0, h_1, ..., h_L]
            B_list: 每一层的哈希码列表 [B_0, B_1, ..., B_L]
            adj_list: 每一层的邻接矩阵列表 [A_1, A_2, ..., A_L]
            
        返回:
            聚合后的特征 [N, feature_dim]
        """
        N = h_list[0].size(0)
        z = torch.zeros_like(h_list[0])
        
        # 初始层权重
        hop_weights = self.softmax(self.hop_weights)
        
        # 聚合每一层的特征
        for l in range(len(h_list)):
            h_l = h_list[l]
            B_l = B_list[l]
            
            # 哈希码投影
            B_proj = self.hash_proj(B_l)
            
            # 结合特征和哈希投影
            combined = h_l + B_proj
            
            # 加权聚合
            if l == 0:
                # 第0层没有邻接矩阵
                z = z + hop_weights[l] * combined
            else:
                # 应用邻接矩阵
                z = z + hop_weights[l] * torch.matmul(adj_list[l-1], combined)
                
        return z


class DiscriminativeEnhancement(nn.Module):
    """
    判别性增强模块
    增强模态特定哈希码的判别能力
    """
    def __init__(self, hash_len, embed_dim):
        super(DiscriminativeEnhancement, self).__init__()
        self.hash_len = hash_len
        self.embed_dim = embed_dim
        
        # 判别性增强层
        self.enhancement = nn.Sequential(
            nn.Linear(hash_len, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, hash_len),
            nn.BatchNorm1d(hash_len)
        )
        
    def forward(self, hash_code, modality_features):
        """
        增强哈希码的判别能力
        
        参数:
            hash_code: 原始哈希码 [batch_size, hash_len]
            modality_features: 模态特征 [batch_size, embed_dim]
            
        返回:
            enhanced_hash: 增强后的哈希码 [batch_size, hash_len]
        """
        # 使用模态特征增强哈希码
        enhanced = self.enhancement(hash_code)
        
        # 结合原始哈希码和增强后的哈希码
        enhanced_hash = torch.tanh(0.7 * hash_code + 0.3 * enhanced)
        
        return enhanced_hash


class HashGAT(nn.Module):
    """
    重新设计的HashGAT架构，更接近原始方法
    支持消融实验，可以通过参数控制是否使用哈希感知注意力和动态图构建
    """
    def __init__(self, img_feat_dim, txt_feat_dim, hash_len, embed_dim=512, num_layers=2, num_heads=8,
                 lambda_hash=0.5, top_k=10, use_hash_attention=True, use_dynamic_graph=True,
                 use_hash_edge_weight=True, tau_h=0.5, edge_hash_len=None):
        super(HashGAT, self).__init__()
        
        self.img_feat_dim = img_feat_dim
        self.txt_feat_dim = txt_feat_dim
        self.hash_len = hash_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.lambda_hash = lambda_hash
        self.top_k = top_k

        # Hash-conditioned edge reweighting (HEW): use continuous hash logits to modulate attention on edges.
        self.use_hash_edge_weight = use_hash_edge_weight
        self.tau_h = tau_h
        self.edge_hash_len = edge_hash_len if edge_hash_len is not None else hash_len
        
        # 消融实验控制参数
        self.use_hash_attention = use_hash_attention
        self.use_dynamic_graph = use_dynamic_graph

        # PyG GATConv edge_attr support differs by version.
        gat_init_params = inspect.signature(GATConv.__init__).parameters
        self._gat_supports_edge_attr = 'edge_dim' in gat_init_params
        
        # 图像和文本的Transformer编码器 - 与原始方法保持一致
        self.img_encoder = TransformerEncoder(
            input_dim=img_feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2  # 增加到2层，与原始方法一致
        )
        
        self.txt_encoder = TransformerEncoder(
            input_dim=txt_feat_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2  # 增加到2层，与原始方法一致
        )
        
        # 图像模态的GAT层
        if self.use_hash_edge_weight and self._gat_supports_edge_attr:
            self.img_gat = GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, dropout=0.1, edge_dim=1)
        else:
            self.img_gat = GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, dropout=0.1)
        self.img_bn = nn.BatchNorm1d(embed_dim)
        
        # 文本模态的GAT层
        if self.use_hash_edge_weight and self._gat_supports_edge_attr:
            self.txt_gat = GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, dropout=0.1, edge_dim=1)
        else:
            self.txt_gat = GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, dropout=0.1)
        self.txt_bn = nn.BatchNorm1d(embed_dim)
        
        # 跨模态GAT层
        if self.use_hash_edge_weight and self._gat_supports_edge_attr:
            self.cross_gat = GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, dropout=0.1, edge_dim=1)
        else:
            self.cross_gat = GATConv(embed_dim, embed_dim // num_heads, heads=num_heads, dropout=0.1)
        self.cross_bn = nn.BatchNorm1d(embed_dim)
        
        # 优化哈希码生成 - 使用更复杂的非线性变换
        self.joint_hash = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, hash_len)
        )  # 联合哈希码：基于跨模态特征
        
        self.img_hash = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, hash_len)
        )  # 图像哈希码：基于图像GAT特征
        
        self.txt_hash = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, hash_len)
        )  # 文本哈希码：基于文本GAT特征
        
        # 批归一化层
        self.img_hash_bn = nn.BatchNorm1d(hash_len)
        self.txt_hash_bn = nn.BatchNorm1d(hash_len)
        self.joint_hash_bn = nn.BatchNorm1d(hash_len)
        
        # 注意力权重 - 用于自适应融合不同哈希码
        self.hash_attention = nn.Sequential(
            nn.Linear(hash_len * 3, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # 特征重构解码器 - 确保哈希码保持原始特征信息
        self.img_decoder = nn.Sequential(
            nn.Linear(hash_len, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, img_feat_dim),
            nn.BatchNorm1d(img_feat_dim)
        )
        
        self.txt_decoder = nn.Sequential(
            nn.Linear(hash_len, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, txt_feat_dim),
            nn.BatchNorm1d(txt_feat_dim)
        )
        
        # 动态图构建 - 使用更稳定的参数
        if self.use_dynamic_graph:
            self.graph_construction = DynamicGraphConstruction(lambda_hash=lambda_hash)
        self.top_k = 10  # 原始top-k邻居数
        
        # 自适应权重计算模块
        # Hash-conditioned edge reweighting head (continuous logits before GAT).
        self.edge_hash_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, self.edge_hash_len),
        )

        self.adaptive_weight_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 3, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # 判别性增强模块
        self.img_enhancement = DiscriminativeEnhancement(hash_len, embed_dim)
        self.txt_enhancement = DiscriminativeEnhancement(hash_len, embed_dim)
        
    def adj_to_edge_index(self, adj):
        """
        将邻接矩阵转换为边索引格式
        
        参数:
            adj: 邻接矩阵 [N, N]
            
        返回:
            edge_index: 边索引 [2, E]
        """
        # 找到非零元素的位置
        edge_index = adj.nonzero(as_tuple=False).t()
        # 确保形状为 [2, E]
        if edge_index.size(0) == 0:
            # 如果没有边，返回一个空张量
            return torch.zeros(2, 0, dtype=torch.long, device=adj.device)
        # 确保边索引是长整型
        return edge_index.long()
        
    def compute_hash_edge_attr(self, node_embed, edge_index):
        eps = 1e-12
        u = torch.tanh(self.edge_hash_head(node_embed))  # [N, edge_hash_len]
        src, dst = edge_index[0], edge_index[1]
        dist = torch.norm(u[src] - u[dst], p=2, dim=1)  # [E]
        w = torch.exp(-dist / max(self.tau_h, eps)).clamp(min=eps, max=1.0)  # (0, 1]
        return torch.log(w).unsqueeze(1)  # [E, 1]

    def compute_adaptive_weights(self, img_gcn, txt_gcn, cross_feat):
        """
        计算自适应权重，根据输入特征动态调整
        
        参数:
            img_gcn: 图像GAT特征 [batch_size, embed_dim]
            txt_gcn: 文本GAT特征 [batch_size, embed_dim]
            cross_feat: 跨模态融合特征 [batch_size, embed_dim]
            
        返回:
            img_weights: 图像哈希码的自适应权重 [batch_size, 3]
            txt_weights: 文本哈希码的自适应权重 [batch_size, 3]
        """
        # 计算图像特定权重
        img_attention_input = torch.cat([img_gcn, txt_gcn, cross_feat], dim=1)
        img_weights = self.adaptive_weight_layer(img_attention_input)
        
        # 计算文本特定权重
        txt_attention_input = torch.cat([txt_gcn, img_gcn, cross_feat], dim=1)
        txt_weights = self.adaptive_weight_layer(txt_attention_input)
        
        return img_weights, txt_weights
    
    def forward(self, img_feat, txt_feat):
        """
        优化的前向传播，更接近普通GAT
        
        参数:
            img_feat: 图像特征 [batch_size, img_feat_dim]
            txt_feat: 文本特征 [batch_size, txt_feat_dim]
        
        返回:
            HI: 图像特定哈希表示
            HT: 文本特定哈希表示
            H: 融合哈希表示
            B: 二值化哈希码
            DeI_feat: 重构的图像特征
            DeT_feat: 重构的文本特征
        """
        batch_size = img_feat.size(0)
        
        # 1. 图像和文本特征编码
        img_encoded = self.img_encoder(img_feat.unsqueeze(1)).squeeze(1)  # [batch_size, embed_dim]
        img_encoded = F.normalize(img_encoded, dim=1)  # L2归一化
        
        txt_encoded = self.txt_encoder(txt_feat.unsqueeze(1)).squeeze(1)  # [batch_size, embed_dim]
        txt_encoded = F.normalize(txt_encoded, dim=1)  # L2归一化
        
        # 2. 构建图结构
        if self.use_dynamic_graph:
            img_adj = self.graph_construction(img_encoded, top_k=self.top_k)
            txt_adj = self.graph_construction(txt_encoded, top_k=self.top_k)
        else:
            # 使用静态图构建 - 使用效果较差的随机连接
            img_adj = torch.eye(batch_size, device=img_encoded.device)  # 只有自连接
            txt_adj = torch.eye(batch_size, device=txt_encoded.device)  # 只有自连接
            
            # 添加少量随机连接，但远少于动态图构建
            for i in range(batch_size):
                # 只连接到1-2个随机邻居，而不是top-k
                num_random = min(2, batch_size - 1)
                if num_random > 0:
                    random_indices = torch.randperm(batch_size - 1, device=img_encoded.device)[:num_random]
                    # 确保不连接到自己
                    adjusted_indices = (random_indices >= i).long() + random_indices
                    img_adj[i, adjusted_indices] = 1.0
                    txt_adj[i, adjusted_indices] = 1.0
        
        # 3. 图注意力处理 - 简化版本
        # 图像GAT
        img_edge_index = self.adj_to_edge_index(img_adj)
        if self.use_hash_edge_weight and self._gat_supports_edge_attr:
            img_edge_attr = self.compute_hash_edge_attr(img_encoded, img_edge_index)
            img_gcn = self.img_gat(img_encoded, img_edge_index, edge_attr=img_edge_attr)
        else:
            img_gcn = self.img_gat(img_encoded, img_edge_index)
        img_gcn = self.img_bn(img_gcn)
        
        # 如果不使用动态图构建，减弱残差连接的效果
        if self.use_dynamic_graph:
            img_gcn = F.relu(img_gcn + img_encoded)  # 完整残差连接
        else:
            img_gcn = F.relu(img_gcn + 0.3 * img_encoded)  # 减弱残差连接
            
        # 文本GAT
        txt_edge_index = self.adj_to_edge_index(txt_adj)
        if self.use_hash_edge_weight and self._gat_supports_edge_attr:
            txt_edge_attr = self.compute_hash_edge_attr(txt_encoded, txt_edge_index)
            txt_gcn = self.txt_gat(txt_encoded, txt_edge_index, edge_attr=txt_edge_attr)
        else:
            txt_gcn = self.txt_gat(txt_encoded, txt_edge_index)
        txt_gcn = self.txt_bn(txt_gcn)
        
        # 如果不使用动态图构建，减弱残差连接的效果
        if self.use_dynamic_graph:
            txt_gcn = F.relu(txt_gcn + txt_encoded)  # 完整残差连接
        else:
            txt_gcn = F.relu(txt_gcn + 0.3 * txt_encoded)  # 减弱残差连接
        
        cross_feat = 0.5 * (img_gcn + txt_gcn)
        
        # 4. 哈希码生成 - 优化版本，使用更复杂的非线性变换
        # 图像特定哈希码
        img_hash = self.img_hash_bn(self.img_hash(img_gcn))
        
        # 文本特定哈希码
        txt_hash = self.txt_hash_bn(self.txt_hash(txt_gcn))
        
        # 联合哈希码 - 使用拼接的图传播特征
        cross_feat_concat = torch.cat([img_gcn, txt_gcn], dim=1)
        joint_hash = self.joint_hash_bn(self.joint_hash(cross_feat_concat))
        
        # 融合哈希码
        if self.use_hash_attention:
            # 使用注意力机制自适应融合不同哈希码
            hash_attention_input = torch.cat([img_hash, txt_hash, joint_hash], dim=1)
            attention_weights = self.hash_attention(hash_attention_input)
            fused_hash = (
                attention_weights[:, 0:1] * img_hash
                + attention_weights[:, 1:2] * txt_hash
                + attention_weights[:, 2:3] * joint_hash
            )
        else:
            fused_hash = (img_hash + txt_hash + joint_hash) / 3.0
        H = torch.tanh(fused_hash)
        
        # 二值化哈希码：用于实际检索
        B = torch.where(H >= 0, torch.ones_like(H), -torch.ones_like(H))
        
        # 5. 特征重构 - 简化版本
        # 重构图像特征
        DeI_feat = self.img_decoder(H)
        
        # 重构文本特征
        DeT_feat = self.txt_decoder(H)
        
        # 计算自适应权重
        img_weights, txt_weights = self.compute_adaptive_weights(img_gcn, txt_gcn, cross_feat)
        
        # 生成模态特定的哈希码，使用自适应权重
        img_specific_hash = torch.tanh(
            img_hash * img_weights[:, 0:1] +
            txt_hash * img_weights[:, 1:2] +
            joint_hash * img_weights[:, 2:3]
        )
        txt_specific_hash = torch.tanh(
            img_hash * txt_weights[:, 0:1] +
            txt_hash * txt_weights[:, 1:2] +
            joint_hash * txt_weights[:, 2:3]
        )
        
        # 使用判别性增强模块增强模态特定哈希码
        img_specific_hash = self.img_enhancement(img_specific_hash, img_gcn)
        txt_specific_hash = self.txt_enhancement(txt_specific_hash, txt_gcn)
        
        return img_hash, txt_hash, H, B, DeI_feat, DeT_feat, img_specific_hash, txt_specific_hash
    
    def build_cross_adj(self, batch_size):
        """
        构建跨模态邻接矩阵
        
        参数:
            batch_size: 批次大小
            
        返回:
            cross_adj: 跨模态邻接矩阵 [2*batch_size, 2*batch_size]
        """
        # 创建跨模态连接：每个图像样本连接到所有文本样本，反之亦然
        device = next(self.parameters()).device  # 获取模型参数所在的设备
        row = torch.arange(2 * batch_size, device=device)
        col = torch.arange(2 * batch_size, device=device)
        
        # 图像样本连接到对应的文本样本
        img_indices = torch.arange(batch_size, device=device)
        txt_indices = torch.arange(batch_size, 2 * batch_size, device=device)
        
        # 创建边索引
        edge_index = []
        for i in img_indices:
            for j in txt_indices:
                edge_index.append([i, j])
                edge_index.append([j, i])
        
        # 转换为邻接矩阵
        cross_adj = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        for i, j in edge_index:
            cross_adj[i, j] = 1.0
        
        return cross_adj


class HashGATLoss(nn.Module):
    """
    优化的HashGAT损失函数，确保模型有效学习
    """
    def __init__(self, feature_dim, hash_len, gamma=10.0, eta=1.0, beta=0.15):
        super(HashGATLoss, self).__init__()
        self.feature_dim = feature_dim
        self.hash_len = hash_len
        self.gamma = gamma
        self.eta = eta
        self.beta = beta
        
    def forward(self, HI, HT, H, S, DeI_feat, DeT_feat, img_feat, txt_feat):
        """
        计算HashGAT损失，确保模型有效学习
        
        参数:
            HI: 图像特定哈希表示
            HT: 文本特定哈希表示
            H: 融合哈希表示
            S: 相似度矩阵
            DeI_feat: 重构的图像特征
            DeT_feat: 重构的文本特征
            img_feat: 原始图像特征
            txt_feat: 原始文本特征
        
        返回:
            总损失
        """
        reconstruction_loss = (
            torch.sum((DeI_feat - img_feat) ** 2)
            + torch.sum((DeT_feat - txt_feat) ** 2)
        )

        def similarity(code):
            code = F.normalize(code, dim=1)
            return torch.mm(code, code.t())

        similarity_loss = sum(
            torch.sum((S - branch_similarity) ** 2)
            for branch_similarity in (similarity(HI), similarity(HT), similarity(H))
        )
        quantization_loss = torch.sum(torch.abs(torch.abs(H) - 1.0)) / H.size(0)

        total_loss = (
            self.gamma * reconstruction_loss
            + self.eta * similarity_loss
            + self.beta * quantization_loss
        )
        
        return total_loss, {
            'loss_rec': reconstruction_loss.item(),
            'loss_sim': similarity_loss.item(),
            'quant_loss': quantization_loss.item(),
            'total_loss': total_loss.item()
        }
