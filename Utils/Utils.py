import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def calcReward(bprLossDiff, keepRate):
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	reward = t.zeros_like(bprLossDiff).cuda()
	reward[posLocs] = 1.0
	return reward

def calcGradNorm(model):
	ret = 0
	for p in model.parameters():
		if p.grad is not None:
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -t.log(nume / deno).mean()


import torch
import numpy as np


def mrr(rank_list, ground_truth):
	for i, item in enumerate(rank_list):
		if item in ground_truth:
			return 1 / (i + 1)
	return 0


def hit_at_k(rank_list, ground_truth, k):
	return len(set(rank_list[:k]) & set(ground_truth)) > 0


def calculate_metrics(scores, ground_truth, topk_list=[1, 3, 10]):
	ranks = torch.argsort(scores, descending=True)
	ranks = ranks.cpu().numpy()
	ground_truth = ground_truth.cpu().numpy() if isinstance(ground_truth, torch.Tensor) else ground_truth

	mrr_score = mrr(ranks, ground_truth)
	hit_scores = [hit_at_k(ranks, ground_truth, k) for k in topk_list]

	return mrr_score, *hit_scores