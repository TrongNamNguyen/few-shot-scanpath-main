import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import common.position_encoding as pe
import torch.nn.functional as F
import numpy as np
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from .backbone.swin import D2SwinTransformer
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .pixel_decoder.fpn import TransformerEncoderPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .config import add_maskformer2_config
from common.position_encoding import get_duration_positional_encoding
import fvcore.nn.weight_init as weight_init



class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2,attn_weights  = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self,
                    tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2,attn_weights = self.self_attn(q,
                              k,
                              value=tgt2,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt, attn_weights

    def forward(self,
                tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask,
                                    query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask,
                                 query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self,
                    tgt,
                    memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self,
                tgt,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self,
                 d_model,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# --- THAY THẾ CLASS ImageFeatureEncoder CŨ BẰNG CLASS MỚI NÀY ---
# (Các class MLP, LayerNorm2d, SelfAttentionLayer, ... giữ nguyên)

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Thêm config của Mask2Former (thay vì add_maskformer2_config cũ)
# Giả sử file config.py của em nằm trong SE-Net/src/config.py
from .config import add_maskformer2_config

class ImageFeatureEncoder(nn.Module):
    def __init__(self,
                 cfg_path,
                 dropout,
                 pixel_decoder='MSD',
                 load_segm_decoder=False,
                 pred_saliency=False):
        super(ImageFeatureEncoder, self).__init__()

        # --- PHẦN SỬA ĐỔI CHÍNH ---
        # Tải mô hình Mask2Former bằng phương pháp chuẩn của Detectron2
        
        cfg = get_cfg()
        add_deeplab_config(cfg) # Cần cho Mask2Former
        add_maskformer2_config(cfg) # File config.py em đã cung cấp
        cfg.merge_from_file(cfg_path) # Tải file resnet50.yaml
        
        # Chúng ta chỉ cần backbone và sem_seg_head (pixel decoder)
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True 
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_model(cfg).sem_seg_head # Tải cả pixel decoder và transformer decoder
        
        # Tải trọng số đã huấn luyện sẵn
        # Chúng ta sẽ tải M2F_R50.pkl cho backbone và M2F_R50_MSDeformAttnPixelDecoder.pkl cho pixel decoder
        # Lưu ý: Các file .pkl em cung cấp dường như là toàn bộ mô hình chứ không phải từng phần.
        # Chúng ta sẽ thử tải M2F_R50.pkl vào toàn bộ mô hình, 
        # vì M2F_R50_MSDeformAttnPixelDecoder.pkl nhỏ hơn, có thể nó chỉ là một phần.
        # Tốt nhất là tải trọng số của toàn bộ mô hình M2F_R50.pkl
        
        # Tải M2F_R50.pkl vào backbone
        bb_weights_path = cfg.MODEL.WEIGHTS # Đường dẫn này sẽ trỏ đến M2F_R50.pkl
        print(f"Đang tải trọng số backbone từ: {bb_weights_path}")
        DetectionCheckpointer(self.backbone, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            bb_weights_path, resume=False
        )
        
        # Tải M2F_R50_MSDeformAttnPixelDecoder.pkl vào pixel_decoder
        # Đường dẫn này sẽ được trỏ từ file config .json
        pd_weights_path = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_WEIGHTS # Thêm 1 key mới vào config
        print(f"Đang tải trọng số pixel decoder từ: {pd_weights_path}")
        # Tải trọng số cho sem_seg_head (bao gồm cả pixel decoder)
        # Chúng ta phải tải thủ công vì nó là một phần của mô hình lớn
        pd_weights = torch.load(pd_weights_path, map_location=torch.device('cpu'))["model"]
        
        # Lọc ra các trọng số chỉ của pixel_decoder (sem_seg_head)
        pixel_decoder_weights = {k.replace("sem_seg_head.", ""): v for k, v in pd_weights.items() if "sem_seg_head." in k}
        self.sem_seg_head.load_state_dict(pixel_decoder_weights, strict=False)
        
        print("Tải trọng số backbone và pixel decoder thành công.")
        # --- KẾT THÚC PHẦN SỬA ĐỔI ---

        self.saliency_head = None # Giữ nguyên logic của em
        if not load_segm_decoder:
            self.segm_decoder = None
        else:
            self.segm_decoder = self.sem_seg_head.predictor # Lấy phần transformer decoder
        
        self.pixel_decoder = self.sem_seg_head.pixel_decoder # Lấy phần pixel decoder (MSDeformAttn)

    def forward(self, x):
        # Chạy backbone
        features = self.backbone(x)
        
        # Chạy pixel decoder
        # Cần 3 features: res3, res4, res5
        # Tên chuẩn của Detectron2: "res2", "res3", "res4", "res5"
        ms_feats_input = {
            "res3": features.get("res3"),
            "res4": features.get("res4"),
            "res5": features.get("res5"),
        }
        
        # Pixel decoder (MSDeformAttn) sẽ xử lý các features này
        # high_res_featmaps là mask_features, ms_feats là transformer_encoder_features
        high_res_featmaps, ms_feats = self.pixel_decoder(ms_feats_input)
        
        # Trả về các features theo đúng định dạng mà UserEmbeddingNet cần
        # high_res_featmaps (từ res2) [bs, 256, 80, 128]
        # ms_feats[0] (từ res5, độ phân giải thấp) [bs, 256, 10, 16]
        # ms_feats[1] (từ res4, độ phân giải trung bình) [bs, 256, 20, 32]
        
        # Lấy feature của res2 làm high_res_featmaps (theo code cũ)
        high_res_featmaps_res2 = self.pixel_decoder.adapter_1(features.get("res2"))

        return high_res_featmaps_res2, ms_feats[0], ms_feats[1] # Trả về s4, s1, s2
# --- HẾT PHẦN THAY THẾ ---

class UserEmbeddingNet(nn.Module):
    def __init__(
        self,
        pa,
        num_decoder_layers: int,
        hidden_dim: int,
        nhead: int,
        ntask: int,
        num_output_layers: int,
        train_encoder: bool = False,
        train_pixel_decoder: bool = False,
        pre_norm: bool = False,
        dropout: float = 0.1,
        dim_feedforward: int = 512,
        num_encoder_layers: int = 3,
    ):
        super(UserEmbeddingNet, self).__init__()
        self.pa = pa
        self.num_decoder_layers = num_decoder_layers
        self.hidden_dim = hidden_dim

        # Encoder: Deformable Attention Transformer
        self.train_encoder = train_encoder
        self.encoder = ImageFeatureEncoder(pa.backbone_config, dropout,
                                           pa.pixel_decoder)
        if not train_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.train_pixel_decoder = train_pixel_decoder
        if train_pixel_decoder:
            self.encoder.pixel_decoder.train()
            for param in self.encoder.pixel_decoder.parameters():
                param.requires_grad = True
        featmap_channels = 256
        if hidden_dim != featmap_channels:
            self.input_proj = nn.Conv2d(featmap_channels,
                                         hidden_dim,
                                         kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()


        # Queries
        # self.ntask = 1
        self.ntask = ntask
        self.aux_queries = 0
        self.query_embed = nn.Embedding(ntask + self.aux_queries, hidden_dim)
        self.query_pos = nn.Embedding(ntask + self.aux_queries, hidden_dim)

        # Decoder
        self.user_fix_cross_attn = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.user_fix_cross_attn.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))

        self.num_encoder_layers = num_encoder_layers
        self.working_memory_encoder = nn.ModuleList(
            [SelfAttentionLayer(
                d_model=self.hidden_dim, nhead=nhead, 
                dropout=dropout, normalize_before=pre_norm
                ) for _ in range(num_encoder_layers)])

        # Positional embedding
        self.pixel_loc_emb = pe.PositionalEncoding2D(pa,
                                                     hidden_dim,
                                                     height=pa.im_h // 4,
                                                     width=pa.im_w // 4,
                                                     dropout=dropout)
        self.pos_scale = 1

        self.fix_ind_emb = nn.Embedding(pa.max_traj_length, hidden_dim)

        # Embedding for distinguishing dorsal or ventral embeddings
        self.dorsal_ind_emb = nn.Embedding(2, hidden_dim)  # P1 and P2
        self.ventral_ind_emb = nn.Embedding(1, hidden_dim)


        # load task embedding    (18, 768)
        self.subject_predictor = MLP(hidden_dim, hidden_dim, pa.num_subjects,
                                         num_output_layers)
        # if self.pa.name == 'CAT2000':
        #     task_emb_dict = np.load('../ISP/CAT2000/GazeformerISP/src/data/CAT2000/embeddings.npy', allow_pickle=True).item()
        # elif self.pa.name == 'COCO-Search18':
        #     task_emb_dict = np.load('../ISP/COCO_Search18/GazeformerISP/src/data/TPTA/embeddings.npy', allow_pickle=True).item()
        # else:
        #     task_emb_dict = np.load('../gazeformer/dataset/embeddings.npy', allow_pickle=True).item()
        # all_task_emb = np.stack([task_emb_dict[key] for key in task_emb_dict], axis=0)
        # self.all_task_emb = torch.tensor(all_task_emb)
        # self.task_transform = nn.Linear(self.all_task_emb.size(-1), hidden_dim)
        # # self.all_task_emb = nn.Embedding(self.ntask, self.hidden_dim)
        # self.cls_embed = nn.Embedding(1, hidden_dim)
        # self.cls_pos = nn.Embedding(1, hidden_dim)
        # self.task_pos = nn.Embedding(1, hidden_dim)
        # self.handcraft_pos = nn.Embedding(5, hidden_dim)

        self.task_transform = nn.Linear(768, hidden_dim)
        self.task_dorsal_encoder = nn.ModuleList()
        for _ in range(self.num_encoder_layers):
            self.task_dorsal_encoder.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dropout=dropout,
                    normalize_before=pre_norm,
                ))
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.cls_pos = nn.Embedding(1, hidden_dim)
        self.task_pos = nn.Embedding(1, hidden_dim)


    def forward(self,
                img: torch.Tensor,
                tgt_seq: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                tgt_seq_high: torch.Tensor,
                duration: torch.Tensor,
                task_emb: torch.Tensor):
        out = {}
        # Prepare dorsal embeddings
        img_embs_s4, img_embs_s1, img_embs_s2 = self.encoder(img)
        high_res_featmaps = self.input_proj(img_embs_s4)
        output_featmaps = high_res_featmaps

        dorsal_embs, dorsal_pos, scale_embs = [], [], []
        # C x 10 x 16
        img_embs = self.input_proj(img_embs_s1)
        bs, c, h, w = img_embs.shape
        pe = self.pixel_loc_emb.forward_featmaps(img_embs.shape[-2:],
                                                    scale=8)
        img_embs = img_embs.view(bs, c, -1).permute(2, 0, 1)
        scale_embs.append(
            self.dorsal_ind_emb.weight[0].unsqueeze(0).unsqueeze(0).expand(
                img_embs.size(0), bs, c))
        dorsal_embs.append(img_embs)
        dorsal_pos.append(
            pe.expand(bs, c, h, w).view(bs, c, -1).permute(2, 0, 1))
        
        dorsal_embs = torch.cat(dorsal_embs, dim=0)
        dorsal_pos = torch.cat(dorsal_pos, dim=0)
        scale_embs = torch.cat(scale_embs, dim=0)

        bs = high_res_featmaps.size(0)


        # Prepare ventral embeddings
        if tgt_seq_high is None:
            tgt_seq = tgt_seq.transpose(0, 1)
            ventral_embs = torch.gather(
                torch.cat([
                    torch.zeros(1, *img_embs.shape[1:],
                                device=img_embs.device), img_embs
                ],
                          dim=0), 0,
                tgt_seq.unsqueeze(-1).expand(*tgt_seq.shape,
                                             img_embs.size(-1)))
            ventral_pos = self.pixel_loc_emb(
                tgt_seq)  # Pos for fixation location
        else:
            tgt_seq_high = tgt_seq_high.transpose(0, 1)
            highres_embs = high_res_featmaps.view(bs, c, -1).permute(2, 0, 1)
            ventral_embs = torch.gather(
                torch.cat([torch.zeros(1,*highres_embs.shape[1:],device=img_embs.device), highres_embs],dim=0), 
                0,
                tgt_seq_high.unsqueeze(-1).expand(*tgt_seq_high.shape, highres_embs.size(-1)))
            # Pos for fixation location
            ventral_pos = self.pixel_loc_emb(tgt_seq_high)

        # Add pos and duration into embeddings
        # Dorsal embeddings
        dorsal_embs += dorsal_pos
        dorsal_pos.fill_(0)
        # Ventral embeddings
        ventral_embs += ventral_pos
        # duration embedding for fixations
        duration_encoding = get_duration_positional_encoding(duration, self.hidden_dim, img_embs.device)
        ventral_pos += duration_encoding
        ventral_pos.fill_(0)
        output_featmaps += self.pixel_loc_emb.forward_featmaps(
            output_featmaps.shape[-2:], scale=self.pos_scale)

        # Add embedding indicator embedding into pos embedding
        dorsal_pos += scale_embs
        ventral_pos += self.ventral_ind_emb.weight.unsqueeze(0).expand(
            *ventral_pos.shape)

        # Temporal embedding for fixations
        # ventral_pos += self.positional_encoding(
        #     ventral_embs).repeat(1, bs, 1)
        ventral_pos += self.fix_ind_emb.weight[:ventral_embs.
                                               size(0)].unsqueeze(1).repeat(
                                                   1, bs, 1)
        ventral_pos[tgt_padding_mask.transpose(0, 1)] = 0
        dorsal_padding = torch.zeros(bs, dorsal_embs.size(0),device=dorsal_embs.device).bool()
        ventral_padding = tgt_padding_mask

        if self.ntask != 1:
            task_emb = self.task_transform(task_emb).unsqueeze(0)
            dorsal_embs = torch.cat((task_emb, dorsal_embs), dim=0)
            dorsal_padding = torch.cat((torch.zeros(bs, 1, device=dorsal_embs.device), dorsal_padding), dim=1)
            task_pos = self.task_pos.weight.unsqueeze(1).repeat(1, bs, 1)
            dorsal_pos = torch.cat((task_pos, dorsal_pos), dim=0)

            for i in range(self.num_encoder_layers):
                # Dorsal cross attention
                dorsal_embs, attn_weights = self.task_dorsal_encoder[
                    i](dorsal_embs,
                    tgt_key_padding_mask=dorsal_padding,
                    query_pos=dorsal_pos)
            out['task_image_weights'] = attn_weights

        
        # Update working memory
        working_memory = torch.cat([dorsal_embs, ventral_embs], dim=0)
        padding_mask = torch.cat([
            torch.zeros(bs, dorsal_embs.size(0),
                        device=dorsal_embs.device).bool(), tgt_padding_mask
        ],
                                    dim=1)

        working_memory_pos = torch.cat([dorsal_pos, ventral_pos], dim=0)

        for i in range(self.num_encoder_layers):
            working_memory, attn_weights = self.working_memory_encoder[i](
                working_memory,
                tgt_key_padding_mask=padding_mask,
                query_pos=working_memory_pos)
            

        dorsal_embs = working_memory[0:dorsal_embs.size(0),:,:]
        dorsal_pos = working_memory_pos[0:dorsal_embs.size(0),:,:]
        ventral_embs = working_memory[-1*self.pa.max_traj_length:,:,:]
        ventral_pos = working_memory_pos[-1*self.pa.max_traj_length:,:,:]


        ventral_padding = tgt_padding_mask
        
        # cross attention with image tokens
        cls_token = self.cls_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # cross attention with fixation tokens
        for i in range(self.num_decoder_layers):
            # Dorsal cross attention
            cls_token, attn_weights = self.user_fix_cross_attn[
                i](cls_token,
                   ventral_embs,
                   memory_mask=None,
                   memory_key_padding_mask=ventral_padding,
                   pos=ventral_pos,
                   query_pos=None)
        
        out['fix_attn_weights'] = attn_weights
        cls_token = cls_token.squeeze(0)
        # output subject_id
        pred_subject_id = self.subject_predictor(cls_token)
        out["pred_subject_id"] = pred_subject_id
        out['user_emb'] = cls_token
        return out
    