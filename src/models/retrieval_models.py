import torch
from torch import nn
import torch.nn.functional as F
from .backbones.CLIP import clip 
from torchvision.models import resnet50 
from transformers import BertModel, RobertaModel, DebertaV2Model
from .backbones.senet import se_resnext50_32x4d
from .backbones.efficientnet import EfficientNet
from .backbones import open_clip

supported_img_encoders = ["open_clip", "CLIP","resnet50","se_resnext50_32x4d","efficientnet-b0","efficientnet-b2","efficientnet-b3", "efficientnet-b4"]
supported_lang_encoders = ["open_clip", "CLIP","BERT"]

'''
For Text-Image Retrieval with Image and Language Encoder are CLIP

'''
class CLIP(nn.Module):
    def __init__(self, model_cfg):
        super(CLIP, self).__init__()

        # visual feature extractor
        self.model_cfg = model_cfg
        self.embed_dim  = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            print(f"====> Using visual backbone: {self.model_cfg.IMG_ENCODER}")
            if self.model_cfg.IMG_ENCODER == "CLIP":
                print(f"====> Using CLIP type: {self.model_cfg.CLIP_TYPE}")
                if self.model_cfg.CLIP_TYPE == "open_clip":
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                else:
                    self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
                self.clip_model = self.clip_model.float()
                self.clip_dim = 512
                self.crop_model = self.clip_model.encode_image
                self.bg_model = self.crop_model	
                self.projection_bg = nn.Linear(self.clip_dim , self.embed_dim)
                self.projection_crop = nn.Linear(self.clip_dim , self.embed_dim)

            elif self.model_cfg.IMG_ENCODER == "resnet50":
                self.crop_model = resnet50(pretrained=False,
                                        num_classes=model_cfg.OUTPUT_SIZE)
                state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                        map_location=lambda storage, loc: storage.cpu())
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
                self.crop_model.load_state_dict(state_dict, strict=False)	
                self.bg_model = self.crop_model
                self.img_in_dim = 1024
                self.projection_bg = nn.Sequential(nn.ReLU(),nn.Linear(self.img_in_dim, self.embed_dim))
                self.projection_crop = nn.Sequential(nn.ReLU(),nn.Linear(self.img_in_dim, self.embed_dim))

            elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.crop_model = se_resnext50_32x4d()
                self.bg_model = self.crop_model	
                self.img_in_dim = 2048
                self.projection_bg = nn.Conv2d(self.img_in_dim, self.embed_dim,kernel_size=1)
                self.projection_crop = nn.Conv2d(self.img_in_dim, self.embed_dim,kernel_size=1)
            
            else:
                self.crop_model = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.bg_model = self.crop_model	
                self.img_in_dim = self.crop_model.out_channels
                self.projection_bg = nn.Linear(self.img_in_dim, self.embed_dim)
                self.projection_crop = nn.Linear(self.img_in_dim, self.embed_dim)
        
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        # texual feature extractor
        if self.model_cfg.LANG_ENCODER in  supported_lang_encoders:
            print(f"====> Using lang backbone: {self.model_cfg.LANG_ENCODER}")
            if self.model_cfg.LANG_ENCODER == "CLIP":
                self.lang_model = self.clip_model.encode_text
                self.projection_lang = nn.Linear(self.clip_dim, self.embed_dim)
                
            elif self.model_cfg.LANG_ENCODER == "BERT":
                print(f"====> Using BERT type: {self.model_cfg.BERT_TYPE}")
                if self.model_cfg.BERT_TYPE == "ROBERTA":
                    self.lang_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
                    self.bert_out_dim = 1024
                    for p in  self.lang_model.parameters():
                        p.requires_grad = False		
                    self.projection_lang = nn.Sequential(nn.LayerNorm(self.bert_out_dim),nn.Linear(self.bert_out_dim, self.bert_out_dim), nn.ReLU(), nn.Linear(self.bert_out_dim, self.embed_dim))

                if model_cfg.BERT_TYPE == "DEBERTA":
                    self.lang_model = DebertaV2Model.from_pretrained(model_cfg.BERT_NAME)
                    self.bert_out_dim = 768
                    for p in  self.lang_model.parameters():
                        p.requires_grad = False		
                    self.projection_lang = nn.Sequential(nn.LayerNorm(self.bert_out_dim),nn.Linear(self.bert_out_dim, self.bert_out_dim), nn.ReLU(), nn.Linear(self.bert_out_dim, self.embed_dim))			

                if self.model_cfg.BERT_TYPE == "BERT":
                    self.lang_model = BertModel.from_pretrained(model_cfg.BERT_NAME)
                    self.bert_out_dim = 768
                    for p in  self.lang_model.parameters():
                        p.requires_grad = False		
                    self.projection_lang = nn.Sequential(nn.Linear(self.bert_out_dim, self.bert_out_dim), nn.ReLU(), nn.Linear(self.bert_out_dim, self.embed_dim))
        else:
            assert self.model_cfg.LANG_ENCODER in supported_lang_encoders, "unsupported lang encoder"

        #projection head 
        self.projection_head_lang = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.projection_head_bg = nn.Sequential(nn.BatchNorm1d(self.embed_dim),nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim//2))
        self.projection_head_crop = nn.Sequential(nn.BatchNorm1d(self.embed_dim),nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim//2))
        self.projection_head_visual = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.BatchNorm1d(self.embed_dim),nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim))
        self.id_cls = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASSES))
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

    # def orthogonal_fusion(self, fg, fl):

    #     fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
    #     fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
    #     fg_norm = torch.norm(fg, dim=1)
        
    #     fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
    #     fl_orth = fl - fl_proj
        
    #     f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
    #     return f_fused

    def visual_forward(self, crop, bg):
        crop_embeds = self.projection_crop(self.crop_model(crop).float())
        crop_embeds = crop_embeds.view(crop_embeds.size(0), -1) 
        crop_head = self.projection_head_crop(crop_embeds)

        bg_embeds = self.projection_bg(self.bg_model(bg).float())
        bg_embeds = bg_embeds.view(bg_embeds.size(0), -1)   
        bg_head = self.projection_head_bg(bg_embeds)

        merge_head = self.projection_head_visual(torch.cat([bg_head,crop_head],dim=-1))
        merge_head = F.normalize(merge_head, p = 2, dim = -1)
# 		print(merge_head.shape)        
        return merge_head  

    def text_forward(self, tokens=None, nl_input_ids=None, nl_attention_mask=None):
        if self.model_cfg.LANG_ENCODER == "CLIP":
            outputs = self.lang_model(tokens).float()
        if self.model_cfg.LANG_ENCODER == "BERT":
            outputs = self.lang_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
            outputs = torch.mean(outputs.last_hidden_state, dim=1)

        lang_head = self.projection_lang(outputs)
        lang_head = self.projection_head_lang(lang_head)
        lang_head = F.normalize(lang_head, p = 2, dim = -1)
# 		print(lang_head.shape)        
        return lang_head
    
    def forward(self, crop, bg, tokens=None, nl_input_ids=None, nl_attention_mask=None):
        visual_head = self.visual_forward(crop,bg)
        lang_head = self.text_forward(tokens, nl_input_ids, nl_attention_mask)
        cls_logits_result = []
        if self.model_cfg.VISUAL_ID_LOSS:
            cls_logits_visual = self.id_cls(visual_head)
            cls_logits_result.append(cls_logits_visual)
        if self.model_cfg.LANG_ID_LOSS:
            cls_logits_lang = self.id_cls(lang_head)
            cls_logits_result.append(cls_logits_lang)
        return [(visual_head,lang_head)],self.logit_scale,cls_logits_result


class CLIP_Extended_Feature_V2(nn.Module):
    def __init__(self, model_cfg):
        super(CLIP_Extended_Feature_V2, self).__init__()

        # visual feature extractor
        self.model_cfg = model_cfg
        self.embed_dim  = self.model_cfg.EMBED_DIM
        if self.model_cfg.IMG_ENCODER in  supported_img_encoders:
            print(f"====> Using visual backbone: {self.model_cfg.IMG_ENCODER}")
            if self.model_cfg.IMG_ENCODER == "CLIP":
                print(f"====> Using CLIP type: {self.model_cfg.CLIP_TYPE}")
                if self.model_cfg.CLIP_TYPE == "open_clip":
                    self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                else:
                    self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
                self.clip_model = self.clip_model.float()
                self.clip_dim = 512
                self.crop_model = self.clip_model.encode_image
                self.bg_model = self.crop_model	
                self.projection_bg = nn.Linear(self.clip_dim , self.embed_dim)
                self.projection_crop = nn.Linear(self.clip_dim , self.embed_dim)

            elif self.model_cfg.IMG_ENCODER == "resnet50":
                self.crop_model = resnet50(pretrained=False,
                                        num_classes=model_cfg.OUTPUT_SIZE)
                state_dict = torch.load(self.model_cfg.RESNET_CHECKPOINT,
                                        map_location=lambda storage, loc: storage.cpu())
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
                self.crop_model.load_state_dict(state_dict, strict=False)	
                self.bg_model = self.crop_model
                self.img_in_dim = 1024
                self.projection_bg = nn.Sequential(nn.ReLU(),nn.Linear(self.img_in_dim, self.embed_dim))
                self.projection_crop = nn.Sequential(nn.ReLU(),nn.Linear(self.img_in_dim, self.embed_dim))

            elif self.model_cfg.IMG_ENCODER == "se_resnext50_32x4d":
                self.crop_model = se_resnext50_32x4d()
                self.bg_model = self.crop_model	
                self.img_in_dim = 2048
                self.projection_bg = nn.Conv2d(self.img_in_dim, self.embed_dim,kernel_size=1)
                self.projection_crop = nn.Conv2d(self.img_in_dim, self.embed_dim,kernel_size=1)
            
            else:
                self.crop_model = EfficientNet.from_pretrained(self.model_cfg.IMG_ENCODER)
                self.bg_model = self.crop_model	
                self.img_in_dim = self.crop_model.out_channels
                self.projection_bg = nn.Linear(self.img_in_dim, self.embed_dim)
                self.projection_crop = nn.Linear(self.img_in_dim, self.embed_dim)
        
        else:
            assert self.model_cfg.IMG_ENCODER in supported_img_encoders, "unsupported img encoder"

        # texual feature extractor
        if self.model_cfg.LANG_ENCODER in  supported_lang_encoders:
            print(f"====> Using lang backbone: {self.model_cfg.LANG_ENCODER}")
            if self.model_cfg.LANG_ENCODER == "CLIP":
                self.lang_model = self.clip_model.encode_text
                self.projection_lang = nn.Linear(self.clip_dim, self.embed_dim)
                self.projection_head_bg_lang = nn.Sequential(nn.LayerNorm(self.clip_dim), nn.ReLU(), nn.Linear(self.clip_dim, self.embed_dim))
                self.projection_head_crop_lang = nn.Sequential(nn.LayerNorm(self.clip_dim), nn.ReLU(), nn.Linear(self.clip_dim, self.embed_dim))
                
            elif self.model_cfg.LANG_ENCODER == "BERT":
                print(f"====> Using BERT type: {self.model_cfg.BERT_TYPE}")
                if self.model_cfg.BERT_TYPE == "ROBERTA":
                    self.lang_model = RobertaModel.from_pretrained(model_cfg.BERT_NAME)
                    self.bert_out_dim = 1024
                    for p in  self.lang_model.parameters():
                        p.requires_grad = False		
                    self.projection_lang = nn.Sequential(nn.LayerNorm(self.bert_out_dim),nn.Linear(self.bert_out_dim, self.bert_out_dim), nn.ReLU(), nn.Linear(self.bert_out_dim, self.embed_dim))

                if model_cfg.BERT_TYPE == "DEBERTA":
                    self.lang_model = DebertaV2Model.from_pretrained(model_cfg.BERT_NAME)
                    self.bert_out_dim = 768
                    for p in  self.lang_model.parameters():
                        p.requires_grad = False		
                    self.projection_lang = nn.Sequential(nn.LayerNorm(self.bert_out_dim),nn.Linear(self.bert_out_dim, self.bert_out_dim), nn.ReLU(), nn.Linear(self.bert_out_dim, self.embed_dim))			

                if self.model_cfg.BERT_TYPE == "BERT":
                    self.lang_model = BertModel.from_pretrained(model_cfg.BERT_NAME)
                    self.bert_out_dim = 768
                    for p in  self.lang_model.parameters():
                        p.requires_grad = False		
                    self.projection_lang = nn.Sequential(nn.Linear(self.bert_out_dim, self.bert_out_dim), nn.ReLU(), nn.Linear(self.bert_out_dim, self.embed_dim))
        else:
            assert self.model_cfg.LANG_ENCODER in supported_lang_encoders, "unsupported lang encoder"

        #projection head 
        self.projection_head_merge_lang = nn.Sequential( nn.Linear(self.embed_dim*2, self.embed_dim*2),nn.LayerNorm(self.embed_dim*2), nn.ReLU(), nn.Linear(self.embed_dim*2, self.embed_dim))

        self.projection_head_merge_visual = nn.Sequential(nn.Linear(self.embed_dim*2, self.embed_dim*2),nn.BatchNorm1d(self.embed_dim*2),nn.ReLU(), nn.Linear(self.embed_dim*2, self.embed_dim))
        self.id_cls = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.BatchNorm1d(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.model_cfg.NUM_CLASSES))
        self.logit_scale = nn.Parameter(torch.ones(()), requires_grad=True)

    def visual_forward(self, crop, bg):
        crop_embeds = self.projection_crop(self.crop_model(crop).float())
        visual_crop_head = crop_embeds.view(crop_embeds.size(0), -1) 

        bg_embeds = self.projection_bg(self.bg_model(bg).float())
        visual_bg_head = bg_embeds.view(bg_embeds.size(0), -1)   

        visual_merge_head = self.projection_head_merge_visual(torch.cat([visual_bg_head,visual_crop_head],dim=-1))
        visual_crop_head, visual_bg_head, visual_merge_head = map(
                    lambda t: F.normalize(t, p=2, dim=-1),
                    (visual_crop_head, visual_bg_head, visual_merge_head))
        visual_heads = [visual_crop_head, visual_bg_head, visual_merge_head]      
        return visual_heads

    def text_forward(self, tokens=None, nl_input_ids=None, nl_attention_mask=None):
        if self.model_cfg.LANG_ENCODER == "CLIP":
            outputs = self.lang_model(tokens).float()
        if self.model_cfg.LANG_ENCODER == "BERT":
            outputs = self.lang_model(nl_input_ids,
                                  attention_mask=nl_attention_mask)
            outputs = torch.mean(outputs.last_hidden_state, dim=1)

        lang_crop_head = self.projection_head_crop_lang(outputs)
        
        lang_bg_head = self.projection_head_bg_lang(outputs)

        lang_merge_head = self.projection_head_merge_lang(torch.cat([lang_bg_head,lang_crop_head],dim=-1))
        
        lang_crop_head, lang_bg_head, lang_merge_head = map(
                    lambda t: F.normalize(t, p=2, dim=-1),
                    (lang_crop_head, lang_bg_head, lang_merge_head))
        lang_heads = [lang_crop_head, lang_bg_head, lang_merge_head]
        return lang_heads
    
    def forward(self, crop, bg, tokens=None, nl_input_ids=None, nl_attention_mask=None):
        visual_heads = self.visual_forward(crop,bg)
        lang_heads = self.text_forward(tokens, nl_input_ids, nl_attention_mask)
        cls_logits_result = []
        pair_heads = []
        visual_cls_embeds, lang_cls_embeds = visual_heads[-1], lang_heads[-1]
        if self.model_cfg.VISUAL_ID_LOSS:
            cls_logits_visual = self.id_cls(visual_cls_embeds)
            cls_logits_result.append(cls_logits_visual)

        if self.model_cfg.LANG_ID_LOSS:
            cls_logits_lang = self.id_cls(lang_cls_embeds)
            cls_logits_result.append(cls_logits_lang)

        for visual_head, lang_head in zip(visual_heads, lang_heads):
            pair_heads.append((visual_head, lang_head))
        return pair_heads, self.logit_scale, cls_logits_result
