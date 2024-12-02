import torchmetrics
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class AssoConcept(pl.LightningModule):
    def init_weight_concept(self):
        self.init_weight = torch.zeros((self.cfg.num_cls, len(self.select_idx)))

        if self.cfg.use_rand_init: 
            torch.nn.init.kaiming_normal_(self.init_weight)
        else: 
            self.init_weight.scatter_(0, self.concept2cls, self.cfg.init_val)
            
        if 'cls_name_init' in self.cfg and self.cfg.cls_name_init != 'none':
            if self.cfg.cls_name_init == 'replace':
                self.init_weight = torch.load(self.init_weight_save_dir)
            elif self.cfg.cls_name_init == 'combine':
                self.init_weight += torch.load(self.init_weight_save_dir)
                self.init_weight = self.init_weight.clip(max=1)
            elif self.cfg.cls_name_init == 'random':
                torch.nn.init.kaiming_normal_(self.init_weight)


    def __init__(self, cfg, init_weight=None, select_idx=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-')))
        concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        select_idx_path = self.data_root.joinpath('select_idx.pth')
        concept_pair_path = self.data_root.joinpath('concept_pair.pth') 
        self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        cls_sim_path = self.data_root.joinpath('cls_sim.pth')

        if concept_feat_path.exists():
            if select_idx is None: self.select_idx = torch.load(select_idx_path)[:cfg.num_concept]
            else: self.select_idx = select_idx

            self.concepts = torch.load(concept_feat_path)[self.select_idx].cuda()
            if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

            self.concept_raw = np.load(concept_raw_path)[self.select_idx]
            
            self.concept2cls = torch.from_numpy(np.load(concept2cls_path)[self.select_idx]).long().view(1, -1)
            if os.path.exists(concept_pair_path):
                self.concept2cls1 = torch.load(concept_pair_path).long()[:,0].view(1, -1)
                self.concept2cls2 = torch.load(concept_pair_path).long()[:,1].view(1, -1)

        if init_weight is None:
            self.init_weight_concept()
        else:
            self.init_weight = init_weight

        if 'cls_sim_prior' in self.cfg and self.cfg.cls_sim_prior and self.cfg.cls_sim_prior != 'none':
            print('use cls prior')
            cls_sim = torch.load(cls_sim_path)
            new_weights = []
            for concept_id in range(self.init_weight.shape[1]):
                target_class = int(torch.where(self.init_weight[:,concept_id] == 1)[0])
                new_weights.append(cls_sim[target_class] + self.init_weight[:,concept_id])
            self.init_weight = torch.vstack(new_weights).T

        self.asso_mat = torch.nn.Parameter(self.init_weight.clone())
        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task="multiclass", average='micro')
        self.valid_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task="multiclass", average='micro')
        self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, task="multiclass", average='micro')
        self.all_y = []
        self.all_pred = []
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.cfg.num_cls, task="multiclass")
        self.save_hyperparameters()


    def _get_weight_mat(self):
        if self.cfg.asso_act == 'relu':
            mat = F.relu(self.asso_mat)
        elif self.cfg.asso_act == 'tanh':
            mat = F.tanh(self.asso_mat) 
        elif self.cfg.asso_act == 'softmax':
            mat = F.softmax(self.asso_mat, dim=-1) 
        else:
            mat = self.asso_mat
        return mat 


    def forward(self, img_feat):
        mat = self._get_weight_mat() # 𝜎(𝑾)
        cls_feat = mat @ self.concepts # 𝑔(𝒙, 𝐶) = 𝒙 ⋅ 𝑬_{𝐶}^T, but it's like 𝜎(𝑾)⋅ 𝑬_{𝐶}^T
        sim = img_feat @ cls_feat.t() # 𝑔(𝒙, 𝐶). 𝜎(𝑾)^T, but it's like 𝒙 ⋅ (𝜎(𝑾)⋅ 𝑬_{𝐶}^T)^T 
        return sim


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch

        sim = self.forward(image)
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if torch.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -torch.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = torch.linalg.vector_norm(F.softmax(self.asso_mat,dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = torch.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        self.log('training_loss', cls_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        final_loss = cls_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return opt

    def validation_step(self, batch, batch_idx):
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        self.log('val_loss', loss)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        all_y = torch.hstack(self.all_y)
        all_pred = torch.hstack(self.all_pred)
        self.total_test_acc = self.test_acc(all_pred, all_y)
        pass

    def on_predict_epoch_start(self):
        self.num_pred = 4
        self.concepts = self.concepts.to(self.device)


    def predict_step(self, batch, batch_idx):
        image, y, image_name = batch
        sim = self.forward(image)
        pred = 100 * sim
        _, y_pred = torch.topk(pred, self.num_pred)
        for img_path, gt, top_pred in zip(image_name, y, y_pred):
            gt = (gt + 1).item()
            top_pred = (top_pred + 1).tolist()
    
    def on_predict_epoch_end(self, results):
        pass
    
    def prune_asso_mat(self, q=0.9, thresh=None):
        asso_mat = self._get_weight_mat().detach()
        val_asso_mat = torch.abs(asso_mat).max(dim=0)[0]
        if thresh is None:
            thresh = torch.quantile(val_asso_mat, q)
        good = val_asso_mat >= thresh
        return good


    def extract_cls2concept(self, cls_names, thresh=0.05):
        asso_mat = self._get_weight_mat().detach()
        strong_asso = asso_mat > thresh 
        res = {}
        import pdb; pdb.set_trace()
        for i, cls_name in enumerate(cls_names): 
            ## threshold globally
            keep_idx = strong_asso[i]
            ## sort
            res[cls_name] = np.unique(self.concept_raw[keep_idx])
        return res


    def extract_concept2cls(self, percent_thresh=0.95, mode='global'):
        asso_mat = self.asso_mat.detach()
        res = {} 
        for i in range(asso_mat.shape[1]):
            res[i] = torch.argsort(asso_mat[:, i], descending=True).tolist()
        return res


class AssoConceptFast(AssoConcept):
    def forward(self, dot_product):
        mat = self._get_weight_mat() # 𝜎(𝑾)
        return dot_product @ mat.t() # 𝑔(𝒙, 𝐶). 𝜎(𝑾)^T
    
class OurModule(nn.Module):
    def __init__(self, dim, num_layers=1):
        super(OurModule, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.LeakyReLU())
        self.linear1 = nn.Sequential(*layers)
        
    def _get_image_embedding(self, original_emb, residual=False, use_img_norm=False):
        A = self.linear1(original_emb) # image
        if use_img_norm:
            A = A / A.norm(dim=-1, keepdim=True)
        if residual:
            A = A+original_emb
        return A

    def forward(self, _A, residual=False, use_img_norm=False):
        A = self._get_image_embedding(_A, residual, use_img_norm) # image
        return A
    
class OurConcept(AssoConcept):
    def __init__(self, cfg, init_weight=None, select_idx=None) -> None:
        super().__init__(cfg, init_weight, select_idx)
        self.attention_block = OurModule(dim=self.concepts.shape[-1], num_layers=self.cfg.num_layers)
        concept2cls_scatter = torch.zeros((self.cfg.num_cls, len(self.select_idx)))
        concept2cls_scatter.scatter_(0, self.concept2cls, 1)
        self.cfg.asso_act=''
        self.mask = self.init_weight.clone().to(device)
        self.asso_mat = torch.nn.Parameter(self.init_weight.clone())
        self.dot_product_bias = torch.nn.Parameter(torch.zeros(len(self.select_idx)))
        self.asso_mat_bias = torch.nn.Parameter(torch.zeros(self.cfg.num_cls))

    def forward(self, image):
        mat = self._get_weight_mat()
        image_embed = self.attention_block(image, 
                                        residual=self.cfg.use_residual, 
                                        use_img_norm=self.cfg.use_img_norm)
        dot_product = image_embed @ self.concepts.transpose(-2, -1)
        dot_product = dot_product + self.dot_product_bias
        pred = dot_product @ mat.t()
        pred = pred + self.asso_mat_bias
        return pred, dot_product
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, 
                            momentum=0.9, weight_decay=1e-4)
        end_factor = self.cfg.lr*1e-2 / self.cfg.lr
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                        end_factor=end_factor, total_iters=self.cfg.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
        
    def _get_weight_mat(self):
        mat = self.asso_mat
        mat = mat*self.mask
        return mat 
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        pred, _ = self.forward(image)
        cls_loss = F.cross_entropy(pred, label)

        if torch.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        self.log('training_loss', cls_loss)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        final_loss = cls_loss
        return final_loss

    def validation_step(self, batch, batch_idx):
        image, y = batch
        pred, _ = self.forward(image)
        loss = F.cross_entropy(pred, y)
        self.log('val_loss', loss)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        pred, _ = self.forward(image)
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss