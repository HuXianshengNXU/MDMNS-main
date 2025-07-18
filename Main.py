import time

import torch
import Utils.TimeLogger as logger
from DataHandler import DataHandler
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
# from DataHandler import DataHandler, rows, cols
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix


class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')

        recallMax = 0
        ndcgMax = 0
        precisionMax = 0
        bestEpoch = 0

        log('Model Initialized')
        total_time = 0
        epoch_times = []
        for ep in range(0, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            start_time = time.perf_counter()
            reses = self.trainEpoch()
            end_time = time.perf_counter()
            epoch_time = end_time - start_time
            total_time += epoch_time
            epoch_times.append(epoch_time)
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if (reses['Recall'] > recallMax):
                    recallMax = reses['Recall']
                    ndcgMax = reses['NDCG']
                    precisionMax = reses['Precision']
                    bestEpoch = ep
                log(self.makePrint('Test', ep, reses, tstFlag))
            print()
        print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision',
              precisionMax)
        average_time = total_time / args.epoch

        print("\n===== 训练耗时统计 =====")
        print(f"总训练时间: {total_time:.4f} 秒")
        print(f"平均每轮耗时: {average_time:.4f} 秒")
        print(f"最快轮次: {min(epoch_times):.4f} 秒")
        print(f"最慢轮次: {max(epoch_times):.4f} 秒")

    def prepareModel(self):
        if args.data == 'tiktok':
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(),
                               self.handler.audio_feats.detach()).cuda()
        else:
            self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()

        out_dims = eval(args.dims) + [args.item]
        in_dims = out_dims[::-1]
        self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
        self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

        out_dims = eval(args.dims) + [args.item]
        in_dims = out_dims[::-1]
        self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
        self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

        if args.data == 'tiktok':
            out_dims = eval(args.dims) + [args.item]
            in_dims = out_dims[::-1]
            self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
            self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def buildUIMatrix(self, u_list, i_list, edge_list):
        mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)
        a = sp.csr_matrix((args.user, args.user))
        b = sp.csr_matrix((args.item, args.item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0

        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epRecLoss, epClLoss = 0, 0, 0
        epDiLoss = 0
        epDiLoss_image, epDiLoss_text = 0, 0
        if args.data == 'tiktok':
            epDiLoss_audio = 0
        steps = trnLoader.dataset.__len__() // args.batch

        diffusionLoader = self.handler.diffusionLoader

        for i, batch in enumerate(diffusionLoader):
            batch_item, batch_index = batch
            batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

            iEmbeds = self.model.getItemEmbeds().detach()
            uEmbeds = self.model.getUserEmbeds().detach()
            # print(iEmbeds.shape)  # torch.Size([7050, 64])
            # print(uEmbeds.shape)  # torch.Size([19445, 64])
            # print("===================")
            # time.sleep(10)

            image_feats = self.model.getImageFeats().detach()
            text_feats = self.model.getTextFeats().detach()
            if args.data == 'tiktok':
                audio_feats = self.model.getAudioFeats().detach()

            self.denoise_opt_image.zero_grad()
            self.denoise_opt_text.zero_grad()
            if args.data == 'tiktok':
                self.denoise_opt_audio.zero_grad()

            if args.data == 'tiktok':
                guide_audio_emb = 0.5 * text_feats + 0.5 * image_feats
                guide_text_emb = 0.5 * image_feats + 0.5 * audio_feats
                guide_image_emb = 0.5 * text_feats + 0.5 * audio_feats

            else :
                guide_text_emb = image_feats
                guide_image_emb = text_feats

            diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(self.denoise_model_image, batch_item,
                                                                                  iEmbeds, batch_index, image_feats,guide_image_emb)
            diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(self.denoise_model_text, batch_item,
                                                                                iEmbeds, batch_index, text_feats,guide_text_emb)
            if args.data == 'tiktok':
                diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(self.denoise_model_audio,
                                                                                      batch_item, iEmbeds, batch_index,
                                                                                      audio_feats,guide_audio_emb)

            loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
            loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
            if args.data == 'tiktok':
                loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss
            epDiLoss_image += loss_image.item()
            epDiLoss_text += loss_text.item()
            if args.data == 'tiktok':
                epDiLoss_audio += loss_audio.item()

            if args.data == 'tiktok':
                loss = loss_image + loss_text + loss_audio
            else:
                loss = loss_image + loss_text

            loss.backward()

            self.denoise_opt_image.step()
            self.denoise_opt_text.step()
            if args.data == 'tiktok':
                self.denoise_opt_audio.step()

            log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

        log('')
        log('Start to re-build UI matrix')

        with torch.no_grad():
            u_list_image = []
            i_list_image = []
            edge_list_image = []

            u_list_text = []
            i_list_text = []
            edge_list_text = []

            if args.data == 'tiktok':
                u_list_audio = []
                i_list_audio = []
                edge_list_audio = []

            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                denoised_batch = self.diffusion_model.p_sample(self.denoise_model_image, batch_item,
                                                               args.sampling_steps, args.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_image.append(int(batch_index[i].cpu().numpy()))
                        i_list_image.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_image.append(1.0)

                denoised_batch = self.diffusion_model.p_sample(self.denoise_model_text, batch_item, args.sampling_steps,
                                                               args.sampling_noise)
                top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list_text.append(int(batch_index[i].cpu().numpy()))
                        i_list_text.append(int(indices_[i][j].cpu().numpy()))
                        edge_list_text.append(1.0)

                if args.data == 'tiktok':
                    denoised_batch = self.diffusion_model.p_sample(self.denoise_model_audio, batch_item,
                                                                   args.sampling_steps, args.sampling_noise)
                    top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

                    for i in range(batch_index.shape[0]):
                        for j in range(indices_[i].shape[0]):
                            u_list_audio.append(int(batch_index[i].cpu().numpy()))
                            i_list_audio.append(int(indices_[i][j].cpu().numpy()))
                            edge_list_audio.append(1.0)
                            
            u_list_image = np.array(u_list_image)
            i_list_image = np.array(i_list_image)
            edge_list_image = np.array(edge_list_image)
            self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
            self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)  # # 边丢弃增强（数据增强）

            u_list_text = np.array(u_list_text)
            i_list_text = np.array(i_list_text)
            edge_list_text = np.array(edge_list_text)
            self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
            self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

            if args.data == 'tiktok':
                u_list_audio = np.array(u_list_audio)
                i_list_audio = np.array(i_list_audio)
                edge_list_audio = np.array(edge_list_audio)
                self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
                self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

        log('UI matrix built!')

        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            self.opt.zero_grad()

            if args.data == 'tiktok':
                usrEmbeds_svt, itmEmbeds_svt, usrEmbeds_st, itmEmbeds_st, usrEmbeds_sv, itmEmbeds_sv,usrEmbeds_sa, itmEmbeds_sa = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix,
                                                             self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds_svt, itmEmbeds_svt, usrEmbeds_st, itmEmbeds_st, usrEmbeds_sv, itmEmbeds_sv = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix,
                                                             self.text_UI_matrix)

            ancEmbeds = usrEmbeds_st[ancs]
            posEmbeds = itmEmbeds_st[poss]
            negEmbeds = itmEmbeds_st[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss1 = - (scoreDiff).sigmoid().log().sum() / args.batch

            ancEmbeds = usrEmbeds_sv[ancs]
            posEmbeds = itmEmbeds_sv[poss]
            negEmbeds = itmEmbeds_sv[negs]
            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss2 = - (scoreDiff).sigmoid().log().sum() / args.batch

            bprLoss = bprLoss1 + bprLoss2

            if args.data == 'tiktok':
                ancEmbeds = usrEmbeds_sa[ancs]
                posEmbeds = itmEmbeds_sa[poss]
                negEmbeds = itmEmbeds_sa[negs]
                scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
                bprLoss3 = - (scoreDiff).sigmoid().log().sum() / args.batch
                bprLoss += bprLoss3

            regLoss = self.model.reg_loss() * args.reg  # L2正则化
            loss = bprLoss + regLoss


            # epRecLoss += bprLoss1.item()
            epRecLoss += bprLoss.item()

            epLoss += loss.item()

            if args.data == 'tiktok':
                usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(
                    self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
            else:
                usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj,
                                                                                          self.image_UI_matrix,
                                                                                          self.text_UI_matrix)

            if args.data == 'tiktok':
                clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2,
                                                                                               poss,
                                                                                               args.temp)) * args.ssl_reg
                clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3,
                                                                                                poss,
                                                                                                args.temp)) * args.ssl_reg
                clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3,
                                                                                                poss,
                                                                                                args.temp)) * args.ssl_reg
            else:
                # clLoss = (contrastLoss(usrEmbeds_st, usrEmbeds_sv, ancs, args.temp) + contrastLoss(itmEmbeds_st, itmEmbeds_sv, poss, args.temp)) * args.ssl_reg

                clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

            def info_nce_loss(z1, z2, temp):
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                self.temperature = temp

                sim_matrix = z1 @ z2.T / self.temperature

                n = z1.size(0)
                labels = torch.arange(n, device=z1.device)

                return F.cross_entropy(sim_matrix, labels)

            def contrastLoss1(embeds1, embeds2, temp):
                embeds1 = F.normalize(embeds1, p=2)
                embeds2 = F.normalize(embeds2, p=2)
                nume = t.exp(t.sum(embeds1 * embeds2, dim=-1) / temp)
                deno = t.exp(embeds1 @ embeds2.T / temp).sum(-1)
                return -t.log(nume / deno).mean()

            @torch.no_grad()
            def add_noise_to_tensor(data):
                noise_levels = [int(args.steps / 10), int(args.steps / 8), int(args.steps / 4), int(args.steps / 2)]
                noisy_tensors = []

                for level in noise_levels:
                    noise = torch.randn_like(data) * level
                    noisy_data = data + noise
                    noisy_tensors.append(noisy_data)

                return noisy_tensors

            neg_itmEmbeds_st = add_noise_to_tensor(itmEmbeds_st)
            neg_itmEmbeds_sv = add_noise_to_tensor(itmEmbeds_sv)

            if args.data == 'tiktok':
                neg_itmEmbeds_sa = add_noise_to_tensor(itmEmbeds_sa)
                loss1 = (info_nce_loss(usrEmbeds_svt, usrEmbeds_st, args.temp) + info_nce_loss(itmEmbeds_svt,
                                                                                               itmEmbeds_st,
                                                                                               args.temp)) * args.ssl_reg
                loss2 = (info_nce_loss(usrEmbeds_svt, usrEmbeds_sv, args.temp) + info_nce_loss(itmEmbeds_svt,
                                                                                               itmEmbeds_sv,
                                                                                               args.temp)) * args.ssl_reg
                loss3 = (info_nce_loss(usrEmbeds_svt, usrEmbeds_sa, args.temp) + info_nce_loss(itmEmbeds_svt,
                                                                                               itmEmbeds_sa,
                                                                                               args.temp)) * args.ssl_reg

                loss4 = 0
                loss5 = 0
                loss6 = 0

                for k in range(len(neg_itmEmbeds_sv)):
                    loss4 += info_nce_loss(itmEmbeds_sv, neg_itmEmbeds_sv[k], args.temp) * args.ssl_reg
                for k in range(len(neg_itmEmbeds_st)):
                    loss5 += info_nce_loss(itmEmbeds_st, neg_itmEmbeds_st[k], args.temp) * args.ssl_reg
                for k in range(len(neg_itmEmbeds_sa)):
                    loss6 += info_nce_loss(itmEmbeds_sa, neg_itmEmbeds_sa[k], args.temp) * args.ssl_reg

                clLoss_ = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            else:

                loss1 = (info_nce_loss(usrEmbeds_svt,usrEmbeds_st,args.temp) + info_nce_loss(itmEmbeds_svt, itmEmbeds_st, args.temp)) * args.ssl_reg
                loss2 = (info_nce_loss(usrEmbeds_svt,usrEmbeds_sv,args.temp) + info_nce_loss(itmEmbeds_svt, itmEmbeds_sv, args.temp)) * args.ssl_reg

                loss3 = 0
                loss4 = 0
                for k in range(len(neg_itmEmbeds_sv)):
                    loss3 += info_nce_loss(itmEmbeds_sv, neg_itmEmbeds_sv[k], args.temp) * args.ssl_reg
                for k in range(len(neg_itmEmbeds_st)):
                    loss4 += info_nce_loss(itmEmbeds_st, neg_itmEmbeds_st[k], args.temp) * args.ssl_reg

                clLoss_ = loss1 + loss2 + loss3 + loss4

            if args.cl_method == 1:
                clLoss = clLoss_

            loss += clLoss

            epClLoss += clLoss.item()

            loss.backward()
            self.opt.step()

            log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
                i,
                steps,
                bprLoss.item(),
                regLoss.item(),
                clLoss.item()
            ), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['BPR Loss'] = epRecLoss / steps
        ret['CL loss'] = epClLoss / steps
        ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
        ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
        if args.data == 'tiktok':
            ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg, epPrecision = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        if args.data == 'tiktok':
            usrEmbeds_svt, itmEmbeds_svt, usrEmbeds_st, itmEmbeds_st, usrEmbeds_sv, itmEmbeds_sv, usrEmbeds_sa, itmEmbeds_sa = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix,
                                                         self.text_UI_matrix, self.audio_UI_matrix)
            # itmEmbeds_svt = itmEmbeds_svt + itmEmbeds_sv + itmEmbeds_st + itmEmbeds_sa
            # usrEmbeds_svt = usrEmbeds_svt + usrEmbeds_sv + usrEmbeds_st + usrEmbeds_sa
        else:
            usrEmbeds_svt, itmEmbeds_svt, usrEmbeds_st, itmEmbeds_st, usrEmbeds_sv, itmEmbeds_sv = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix,
                                                         self.text_UI_matrix)

            # itmEmbeds_svt = itmEmbeds_svt + itmEmbeds_sv + itmEmbeds_st
            # usrEmbeds_svt = usrEmbeds_svt + usrEmbeds_sv + usrEmbeds_st


        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            allPreds = torch.mm(usrEmbeds_st[usr], torch.transpose(itmEmbeds_st, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            allPreds += torch.mm(usrEmbeds_sv[usr], torch.transpose(itmEmbeds_sv, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            if args.data == 'tiktok' :
                allPreds += torch.mm(usrEmbeds_sa[usr], torch.transpose(itmEmbeds_sa, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            allPreds += torch.mm(usrEmbeds_svt[usr], torch.transpose(itmEmbeds_svt, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = torch.topk(allPreds, args.topk)
            recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            epPrecision += precision
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision),
                save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        ret['Precision'] = epPrecision / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = allPrecision = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = precision = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
                    precision += 1
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            precision = precision / args.topk
            allRecall += recall
            allNdcg += ndcg
            allPrecision += precision
        return allRecall, allNdcg, allPrecision


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


if __name__ == '__main__':
    seed_it(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    coach.run()
