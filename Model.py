import time

import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))  # 用户嵌入
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))  # 物品嵌入
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])  # GCN层，对应图传播

		self.edgeDropper = SpAdjDropEdge(args.keepRate)  # 边丢弃数据增强，对应论文数据增强策略

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)  # 视觉特征参数矩阵
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)  # 文本特征参数矩阵
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)  # 音频特征线性转换（TikTok专用）
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		# 存储原始多模态特征
		self.image_embedding = image_embedding  # 视觉特征
		self.text_embedding = text_embedding  # 文本特征
		if audio_embedding != None:
			self.audio_embedding = audio_embedding  # 音频特征
		else:
			self.audio_embedding = None

		# 模态权重参数，对应论文2.4节多模态聚合权重学习
		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))  # 三模态均匀初始化
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))  # 两模态均匀初始化
		self.softmax = nn.Softmax(dim=0)  # 权重归一化

		self.dropout = nn.Dropout(p=0.1)  #  dropout层

		self.leakyrelu = nn.LeakyReLU(0.2)  # 激活函数
				
	def getItemEmbeds(self):  # 获取物品基础嵌入
		return self.iEmbeds
	
	def getUserEmbeds(self):  # 获取用户基础嵌入
		return self.uEmbeds
	
	def getImageFeats(self):
		"""视觉特征转换函数，对应论文2.3.1节模态特征对齐"""
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))  # 参数矩阵转换
			return image_feats
		else:
			return self.image_trans(self.image_embedding)  # 线性层转换
	
	def getTextFeats(self):
		"""文本特征转换函数"""
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)

	def getAudioFeats(self):
		"""音频特征转换函数（TikTok专用）"""
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):

		"""多模态图聚合主函数，对应论文2.4节Multi-Modal Graph Aggregation"""

		# --------------------- 多模态特征转换 ---------------------
		# 处理视觉和文本特征，根据配置选择参数矩阵或线性层转换
		# print("1adj.shape==",adj.shape)
		if args.trans == 0:  # 使用参数矩阵转换
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:  # 使用线性层转换
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:   # 混合模式（视觉用参数矩阵，文本用线性层）
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		# 处理音频特征（仅TikTok数据集）
		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		# --------------------- 模态权重学习 ---------------------
		weight = self.softmax(self.modal_weight)  # 对模态权重归一化，对应论文2.4节模态加权聚合

		# --------------------- 视觉模态图聚合 ---------------------
		# 1. 使用扩散生成的视觉模态图（image_adj）进行消息传递
		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])  # 拼接用户和物品基础嵌入
		# print(image_adj.shape)  # torch.Size([26495, 26495])
		# print(embedsImageAdj.shape)  # torch.Size([26495, 64])
		# print(self.iEmbeds.shape)  # torch.Size([7050, 64])
		# time.sleep(100)
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)  # 稀疏矩阵乘法，对应扩散生成图的一阶邻居聚合 相当于模态嵌入-----------------------
		# print(embedsImageAdj.shape)  # torch.Size([26495, 64])

		# 2. 使用原始交互图（adj）进行多模态特征聚合
		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])  # 拼接用户嵌入和归一化视觉特征
		embedsImage = torch.spmm(adj, embedsImage)  # 原始图消息传递，捕获用户-物品交互与视觉特征的关联

		# print(embedsImage.shape)  # torch.Size([26495, 64])
		# time.sleep(100)

		# 3. 二阶邻居聚合（用户-物品-物品交互）
		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)
		embedsImage += embedsImage_  # 相当于结构+视觉-----------------------------------------------------------------
		# print(embedsImage.shape)  # torch.Size([26495, 64])
		# print("-----------------------")
		# time.sleep(10)



		# --------------------- 文本模态图聚合（逻辑同视觉模态）---------------------
		# 文本模态图聚合
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		# print(embedsText.shape)  # torch.Size([26495, 64])
		# print(F.normalize(text_feats).shape)  # torch.Size([7050, 64])
		# print(self.uEmbeds.shape)  # torch.Size([19445, 64])
		# time.sleep(100)
		embedsText = torch.spmm(adj, embedsText)

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)
		embedsText += embedsText_  # 相当于结构+文本----------------------------------------------------------

		# --------------------- 音频模态图聚合（仅TikTok）---------------------
		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		# --------------------- 融合扩散图与原始图信号 ---------------------
		# 通过超参数 ris_adj_lambda 控制扩散图信号的权重，对应论文中模态感知图与协作信号的融合     模态与结构+模态再次融入，得到增强的结构+模态---------------------------
		embedsImage += args.ris_adj_lambda * embedsImageAdj
		embedsText += args.ris_adj_lambda * embedsTextAdj
		# print(embedsImage.shape)  # torch.Size([26495, 64])
		# print(embedsText.shape)  # torch.Size([26495, 64])
		# print("=====================")
		# time.sleep(10)

		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj

		# --------------------- 多模态加权聚合 ---------------------
		# print(embedsImage.shape)  # torch.Size([26495, 64])
		# print(embedsText.shape)  # torch.Size([26495, 64])
		# print("-----")
		# time.sleep(100)
		# print(adj.shape)  # torch.Size([26495, 26495]) 原始UI图
		# time.sleep(100)

		# 两模态（视觉+文本）加权求和
		if audio_adj == None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText  # 两模态（视觉+文本）加权求和
		else:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio  # 三模态（视觉+文本+音频）加权求和

		# print(embedsModal.shape)  # torch.Size([26495, 64])
		# time.sleep(100)

		# 文本模态图聚合
		embedsModalFusedAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		# print(embedsModal.shape)
		# print(embedsModalFusedAdj.shape)
		embedsModalFusedAdj = torch.spmm(adj, embedsModalFusedAdj)

		embedsModalFused = torch.concat([self.uEmbeds, F.normalize(embedsModal[args.user:])]) # embedsText[:args.user]
		# print("--------------------------------")
		# print(F.normalize(embedsModal).shape)  # torch.Size([26495, 64])
		# print(self.uEmbeds.shape)  # torch.Size([19445, 64])
		# print(embedsModalFused.shape)  # torch.Size([45940, 64])
		# # print(adj.shape)
		# print("2adj.shape==", adj.shape)
		embedsModalFused = torch.spmm(adj, embedsModalFused)

		embedsModalFused_ = torch.concat([embedsModalFused[:args.user], self.iEmbeds])
		embedsModalFused_ = torch.spmm(adj, embedsModalFused_)
		embedsModalFused += embedsModalFused_

		embedsModalFused += args.ris_adj_lambda * embedsModalFusedAdj

		# embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsModalFused
		# print(embedsModalFused.shape)
		#
		# print("结构+文本+视觉")
		# time.sleep(100)

		# --------------------- GCN层高阶特征提取 ---------------------
		# embeds = embedsModal
		# if audio_adj != None:
		svt_embedsModal = embedsModalFused
		svt_embeds = svt_embedsModal
		svt_embedsLst = [svt_embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, svt_embedsLst[-1])  # 逐层图卷积，捕获高阶协作关系
			svt_embedsLst.append(embeds)
		svt_embeds = sum(svt_embedsLst)  # 多层聚合，对应论文中"sum-pooling"操作
		svt_embeds = svt_embeds + args.ris_lambda * F.normalize(svt_embedsModal)
		# print(svt_embeds.shape)
		# time.sleep(100)

		st_embedsModal = embedsText
		st_embeds = st_embedsModal
		st_embedsLst = [st_embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, st_embedsLst[-1])  # 逐层图卷积，捕获高阶协作关系
			st_embedsLst.append(embeds)
		st_embeds = sum(st_embedsLst)  # 多层聚合，对应论文中"sum-pooling"操作
		st_embeds = st_embeds + args.ris_lambda * F.normalize(st_embedsModal)

		sv_embedsModal = embedsImage
		sv_embeds = sv_embedsModal
		sv_embedsLst = [sv_embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, sv_embedsLst[-1])  # 逐层图卷积，捕获高阶协作关系
			sv_embedsLst.append(embeds)
		sv_embeds = sum(sv_embedsLst)  # 多层聚合，对应论文中"sum-pooling"操作
		sv_embeds = sv_embeds + args.ris_lambda * F.normalize(sv_embedsModal)

		# embeds = embedsModal
		# embedsLst = [embeds]
		# for gcn in self.gcnLayers:
		# 	embeds = gcn(adj, embedsLst[-1])  # 逐层图卷积，捕获高阶协作关系
		# 	embedsLst.append(embeds)
		# embeds = sum(embedsLst)  # 多层聚合，对应论文中"sum-pooling"操作

		if audio_adj != None:
			sa_embedsModal = embedsAudio
			sa_embeds = sa_embedsModal
			sa_embedsLst = [sa_embeds]
			for gcn in self.gcnLayers:
				embeds = gcn(adj, sa_embedsLst[-1])  # 逐层图卷积，捕获高阶协作关系
				sa_embedsLst.append(embeds)
			sa_embeds = sum(sa_embedsLst)  # 多层聚合，对应论文中"sum-pooling"操作
			sa_embeds = sa_embeds + args.ris_lambda * F.normalize(sa_embedsModal)

		# --------------------- 残差连接与正则化 ---------------------
		# embeds = embeds + args.ris_lambda * F.normalize(embedsModal)
		if audio_adj == None:
			return svt_embeds[:args.user], svt_embeds[args.user:], st_embeds[:args.user], st_embeds[args.user:], sv_embeds[:args.user], sv_embeds[args.user:]
		else :
			return svt_embeds[:args.user], svt_embeds[args.user:], st_embeds[:args.user], st_embeds[args.user:], sv_embeds[:args.user], sv_embeds[args.user:],sa_embeds[:args.user], sa_embeds[args.user:]
		# return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		# print(type(adj), type(image_adj), type(text_adj), type(audio_adj))
		# time.sleep(10)
		"""跨模态对比学习视图生成函数，对应论文2.3节Cross-Modal Contrastive Augmentation"""

		# --------------------- 多模态特征转换（同forward_MM）---------------------
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		# --------------------- 生成各模态对比视图 ---------------------
		# 视觉模态视图：基于扩散生成的视觉图（image_adj）聚合
		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		# print(F.normalize(image_feats).shape)
		# time.sleep(100)
		embedsImage = torch.spmm(image_adj, embedsImage)

		# 文本模态视图：基于扩散生成的文本图（text_adj）聚合
		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		# 音频模态视图（仅TikTok）
		if audio_adj != None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		# --------------------- GCN层特征提取（对比视图增强）---------------------
		# 视觉视图的高阶特征提取
		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])  # 通过原始图传播增强视图的协作信息
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)  # 多层聚合

		# 文本视图的高阶特征提取（逻辑同上）
		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		# 音频视图的高阶特征提取（仅TikTok）
		if audio_adj != None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		# --------------------- 返回对比视图 ---------------------
		if audio_adj == None:
			# 返回视觉和文本视图（用户和物品嵌入）
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		else:
			# 返回视觉、文本、音频视图（用户和物品嵌入）
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	"""边丢弃数据增强层，对应论文数据增强策略"""
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate  # 保留边的比例

	def forward(self, adj):
		"""随机丢弃边并重新归一化"""
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)  # 伯努利采样

		newVals = vals[mask] / self.keepRate  # 权重重缩放
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
		
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

		# print(self.betas.shape)
		# time.sleep(1000)



	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False):
		# 去噪
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		# self.model = GCNRecommender(x.shape[0], x.shape[1], args.latdim).cuda()
		# fused_output = self.model(x, guide_emb)
		model_output = model(x, t, False)

		# print(x.shape)
		# time.sleep(1000)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats,guide_emb):
		batch_size = x_start.size(0)

		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start
		# print(x_t)
		# print(guide_emb)
		# time.sleep(1000)
		self.model = GCNRecommender(x_t.shape[0], x_t.shape[1], args.latdim).cuda()
		fused_output = self.model(x_t, guide_emb)
		# print(fused_output.shape)
		# print(model_feats.shape)
		# print(x_t.shape)
		# print(guide_emb.shape)
		# time.sleep(100)
		# x_t += guide_emb
		# model_output = model(x_t, ts)
		# print(fused_output)
		# time.sleep(1000)
		# # print(ts)

		model_output = model(fused_output, ts)

		# print("success")
		# print(model_output.shape)
		# print("-----------------------")
		# time.sleep(100)
		# print()
		mse = self.mean_flat((x_start - model_output) ** 2)
		# print(mse)
		# print(x_start)
		# print(model_output)
		# time.sleep(1000)
		# print(x_start.shape)  # torch.Size([1024, 6710])
		# print(guide_emb.shape)  # torch.Size([6710, 64])
		# time.sleep(1000)
		self.mlp = nn.Linear(guide_emb.shape[-1],x_start.shape[0]).cuda()
		guide_emb_result = self.mlp(guide_emb)
		denoise_mse = self.mean_flat((x_start - guide_emb_result.T) ** 2)
		# print(mse.shape)
		# print(denoise_mse.shape)
		# time.sleep(1000)
		# print(mse)
		# print(denoise_mse)
		# time.sleep(1000)
		mse = mse + denoise_mse


		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)

		diff_loss = weight * mse
		# print(diff_loss)
		# time.sleep(1000)
		# print(weight)
		# print(mse)
		# time.sleep(1000)

		usr_model_embeds = torch.mm(model_output, model_feats)
		usr_id_embeds = torch.mm(x_start, itmEmbeds)

		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

		return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


class GCNLayer1(nn.Module):
	def __init__(self, in_features, out_features):
		super(GCNLayer1, self).__init__()
		self.linear = nn.Linear(in_features, out_features)

	def forward(self, adj, features):
		"""
        参数:
            adj: 邻接矩阵 (num_nodes, num_nodes)
            features: 特征矩阵 (num_nodes, in_features)
        返回:
            output: 聚合后的特征 (num_nodes, out_features)
        """
		support = torch.mm(adj, features)  # 邻接矩阵乘以特征矩阵
		output = self.linear(support)  # 线性变换
		return output


class GCNRecommender(nn.Module):
	def __init__(self, num_users, num_items, item_features_dim, hidden_dim=args.guide_fused_hidden_dim):
		super(GCNRecommender, self).__init__()
		# User-Item交互矩阵的嵌入层
		self.user_embedding = nn.Embedding(num_users, hidden_dim)
		self.item_embedding = nn.Embedding(num_items, hidden_dim)

		# Item-Feature特征转换层
		self.feature_transform = nn.Linear(item_features_dim, hidden_dim)

		# GCN层
		self.gcn_layer = GCNLayer1(hidden_dim, hidden_dim)

	def forward(self, user_item_matrix, item_feature_matrix):
		"""
        参数:
            user_item_matrix: 用户-物品交互矩阵 (num_users, num_items)
            item_feature_matrix: 物品-特征矩阵 (num_items, item_features_dim)
        返回:
            user_item_embedding: 融合后的用户-物品表示 (num_users, num_items)
        """
		num_users, num_items = user_item_matrix.size()

		# 构建邻接矩阵 (用户-物品二分图)
		# [ [0,   user_item_matrix],
		#   [user_item_matrix^T, 0] ]
		adj_top = torch.cat([
			torch.zeros(num_users, num_users, device=user_item_matrix.device),
			user_item_matrix
		], dim=1)

		adj_bottom = torch.cat([
			user_item_matrix.t(),
			torch.zeros(num_items, num_items, device=user_item_matrix.device)
		], dim=1)

		adj_matrix = torch.cat([adj_top, adj_bottom], dim=0)

		# 归一化邻接矩阵
		adj_norm = self.normalize_adj(adj_matrix)

		# 初始化节点特征
		user_embed = self.user_embedding.weight
		item_embed = self.item_embedding.weight
		item_features = self.feature_transform(item_feature_matrix)

		# 合并用户和物品特征
		node_features = torch.cat([user_embed, item_features], dim=0)

		# GCN传播
		gcn_output = self.gcn_layer(adj_norm, node_features)

		# 分离用户和物品表示
		user_output = gcn_output[:num_users, :]
		item_output = gcn_output[num_users:, :]

		# 计算用户-物品交互分数
		user_item_embedding = torch.mm(user_output, item_output.t())

		return user_item_embedding

	# def normalize_adj(self, adj):
	# 	"""对称归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)"""
	# 	# 添加自环
	# 	adj_plus_eye = adj + torch.eye(adj.size(0), device=adj.device)
	#
	# 	# 计算度矩阵
	# 	row_sum = adj_plus_eye.sum(1)
	# 	d_inv_sqrt = torch.pow(row_sum, -0.5)
	# 	d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
	# 	d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
	#
	# 	# 归一化
	# 	normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_plus_eye), d_mat_inv_sqrt)
	# 	return normalized_adj

	def normalize_adj(self, adj):
		"""对称归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)"""
		# 添加自环
		adj_plus_eye = adj + torch.eye(adj.size(0), device=adj.device)

		# 计算度矩阵
		row_sum = adj_plus_eye.sum(1)
		# 避免除零错误
		# print(row_sum)
		# time.sleep(1000)
		row_sum = torch.clamp(row_sum, min=1e-8)
		d_inv_sqrt = torch.pow(row_sum, -0.5)
		d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

		# 归一化
		normalized_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_plus_eye), d_mat_inv_sqrt)
		return normalized_adj