import mmcv
from mmcv import Config
import os.path as osp
import pickle
import torch
import tqdm
import numpy as np
import clip
from PIL import Image

def pre_exp(cfg_file, work_dir):
    """
    Load the config from cfg_file, create a folder work_dir to save everything. 
    The config file will be saved to work_dir as well. 
    """
    cfg = Config.fromfile(cfg_file)
    mmcv.mkdir_or_exist(work_dir)
    cfg.dump(osp.join(work_dir, osp.basename(cfg_file)))
    cfg.work_dir = work_dir
    return cfg


"""
Helper function for Pickle, to avoid with context
"""
def pickle_load(f_name):
    try:
        with open(f_name, 'rb') as f:
            return pickle.load(f)
    except:
        print(f_name)
        raise RuntimeError('cannot load file')


def pickle_dump(obj, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(obj, f)


"""
Data preprocessing function, to process data in a batch fashion
"""
def batchify_run(process_fn, data_lst, res, batch_size, use_tqdm=False):
    data_lst_len = len(data_lst)
    num_batch = np.ceil(data_lst_len / batch_size).astype(int)
    iterator = range(num_batch)
    if use_tqdm:
        iterator = tqdm.tqdm(iterator)
    for i in iterator:
        batch_data = data_lst[i * batch_size:(i + 1) * batch_size]
        batch_res = process_fn(batch_data)
        res[i * batch_size:(i + 1) * batch_size] = batch_res
        del batch_res


def prepare_img_feat_finetuned_model(img_names,
                pretrained_model,
                ckpt_path=None,
                save_path=None,
                clip_model_name='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load(clip_model_name, device=device)

    if clip_model_name == 'ViT-B/32' or clip_model_name == 'ViT-B/16':
        latent_dim = 512
    elif clip_model_name == 'ViT-L/14':
        latent_dim = 768
    elif clip_model_name=='RN50':
        latent_dim = 1024
    elif clip_model_name=='RN101':
        latent_dim = 512

    res = torch.empty((len(img_names), latent_dim))
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    def process_img(img_names):
        img_tensor = torch.cat([preprocess(Image.open('{}'.format(img_name)))\
                .unsqueeze(0).to(device) \
                for img_name in img_names])
        with torch.no_grad():
            img_feat = pretrained_model.attention_block._get_image_embedding(img_tensor)
        return img_feat
    
    batchify_run(process_img, img_names, res, 2048, use_tqdm=True)
    if save_path:
        torch.save(res, save_path)
    return res

def prepare_img_feat(img_names,
                    ckpt_path=None,
                    save_path=None,
                    clip_model_name='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load(clip_model_name, device=device)

    if clip_model_name == 'ViT-B/32' or clip_model_name == 'ViT-B/16':
        latent_dim = 512
    elif clip_model_name == 'ViT-L/14':
        latent_dim = 768
    elif clip_model_name=='RN50':
        latent_dim = 1024
    elif clip_model_name=='RN101':
        latent_dim = 512

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    def process_img(img_names):
        img_tensor = torch.cat([preprocess(Image.open('{}'.format(img_name)))\
                .unsqueeze(0).to(device) \
                for img_name in img_names])
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
        return img_feat

    res = torch.empty((len(img_names), latent_dim))
    batchify_run(process_img, img_names, res, 512, use_tqdm=True)
    if save_path:
        torch.save(res, save_path)
    return res


def prepare_img_feat_from_processed(img_names,
                                    ckpt_path=None,
                                    save_path=None,
                                    latent_dim=512,
                                    clip_model_name='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("{}".format(clip_model_name), device=device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
    res = torch.empty((len(img_names), latent_dim))

    def process_img(img_names):
        img_lst = [pickle_load(img_name) for img_name in img_names]
        print('pickle load')
        img_tensor = torch.stack(img_lst)
        print('form tensor')
        img_tensor = img_tensor.to(device)
        print('to gpu')
        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
        return img_feat

    batchify_run(process_img, img_names, res, 2048, use_tqdm=True)
    if save_path:
        torch.save(res, save_path)
    return res


def prepare_txt_feat(prompts, ckpt_path=None, save_path=None, clip_model_name='ViT-B/32'):
    if clip_model_name == 'ViT-B/32' or clip_model_name == 'ViT-B/16':
        latent_dim = 512
    elif clip_model_name == 'ViT-L/14':
        latent_dim = 768
    elif clip_model_name=='RN50':
        latent_dim = 1024
    elif clip_model_name=='RN101':
        latent_dim = 512
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load("{}".format(clip_model_name), device=device)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
    

    def process_txt(prompts):
        token = torch.cat([clip.tokenize(prompt)
                        for prompt in prompts]).to(device)
        with torch.no_grad():
            txt_feat = model.encode_text(token)
        return txt_feat
        
    
    res = torch.empty((len(prompts), latent_dim))
    batchify_run(process_txt, prompts, res, 128, use_tqdm=True)
    if save_path:
        torch.save(res, save_path)
    return res

def prepare_txt_token(prompts, ckpt_path=None, save_path=None, latent_dim=77,clip_model_name='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("{}".format(clip_model_name), device=device)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])

    res = torch.empty((len(prompts), latent_dim))

    def process_txt(prompts):
        token = torch.cat([clip.tokenize(prompt)
                           for prompt in prompts]).to(device)
        return token

    batchify_run(process_txt, prompts, res, 2048, use_tqdm=True)
    if save_path:
        torch.save(res, save_path)
    return res