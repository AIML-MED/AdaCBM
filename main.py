import utils
import pytorch_lightning as pl
import torch
import numpy as np
import json
import numpy as np
import os
from datetime import datetime
import argparse
from lightning.pytorch.loggers import TensorBoardLogger

device = "cuda" if torch.cuda.is_available() else "cpu"

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

def save_npy_files(class2concepts, save_dir):
    # sort class name to make sure they are in the same order, to avoid potential problem
    class_names = sorted(list(class2concepts.keys()))
    num_concept = sum([len(concepts) for concepts in class2concepts.values()])
    concept2cls = np.zeros(num_concept)
    i = 0
    all_concepts = []
    for class_name, concepts in class2concepts.items():
        class_idx = class_names.index(class_name)
        for concept in concepts:
            all_concepts.append(concept)
            concept2cls[i] = class_idx
            i += 1
    np.save(save_dir + 'concepts_raw.npy', np.array(all_concepts))
    np.save(save_dir + 'cls_names.npy', np.array(class_names))
    np.save(save_dir + 'concept2cls.npy', concept2cls)

def our_main(cfg):
    start_time = datetime.now()
    from models.asso_opt.asso_our import OurConcept
    from models.select_concept.select_algo import our_selection
    from data import DataModule
    import random
    
    MODEL = OurConcept

    cfg.concept2type = False
    cfg.use_txt_norm = False
    cfg.use_img_norm = False
    cfg.use_residual = False
    
    if type(cfg.seed)==list:
        cfg.seed = int(cfg.seed[0])
    if type(cfg.concept_select_fn)==list:
        cfg.concept_select_fn = cfg.concept_select_fn[0]
    if type(cfg.num_concept)==list:
        cfg.num_concept = int(cfg.num_concept[0])
    
    print('Configurations:', cfg)
    print('Model:', MODEL.__name__)

    class2concepts = json.load(open(cfg.concept_path, "r"))

    save_npy_files(class2concepts, cfg.concept_root)
    
    if not cfg.seed: cfg.seed = 123
    pl.seed_everything(int(cfg.seed))
    
    concept_select_fn = our_selection

    data_module = DataModule(
        cfg.num_concept,
        cfg.data_root,
        cfg.clip_model,
        cfg.img_split_path,
        cfg.img_path,
        cfg.n_shots,
        cfg.raw_sen_path,
        cfg.concept2cls_path,
        concept_select_fn,
        cfg.cls_name_path,
        cfg.bs,
        on_gpu=cfg.on_gpu,
        num_workers=cfg.num_workers if 'num_workers' in cfg else 0,
        img_ext=cfg.img_ext if 'img_ext' in cfg else '.jpg',
        clip_ckpt=cfg.ckpt_path if 'ckpt_path' in cfg else None,
        use_txt_norm=cfg.use_txt_norm if 'use_txt_norm' in cfg else False, 
        use_img_norm=cfg.use_img_norm if 'use_img_norm' in cfg else False,
        use_cls_name_init=cfg.cls_name_init if 'cls_name_init' in cfg else 'none',
        use_cls_sim_prior=cfg.cls_sim_prior if 'cls_sim_prior' in cfg else 'none',
        remove_cls_name=cfg.remove_cls_name if 'remove_cls_name' in cfg else True,
        concept2type=cfg.concept2type,
        pearson_weight=cfg.pearson_weight if 'pearson_weight' in cfg else None
    )

    model = MODEL(cfg, init_weight=torch.load(cfg.init_weight_path) if 'init_weight_path' in cfg else None)

    check_interval = 1
    
    if cfg.test:
        ckpt_path = cfg.ckpt_path
        print('load ckpt: {}'.format(ckpt_path))
        model = MODEL.load_from_checkpoint(str(ckpt_path))
        trainer = pl.Trainer()
        trainer.test(model, data_module)
        # test_acc = round(100 * float(model.total_test_acc), 2)
        return

    print("check interval = {}".format(check_interval))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.work_dir,
        filename='{epoch}-{step}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
        every_n_epochs=check_interval)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    tb_logger = TensorBoardLogger(save_dir=cfg.work_dir)
    trainer = pl.Trainer(max_epochs=int(cfg.max_epochs) if 'max_epochs' in cfg else 1000,
        callbacks=[checkpoint_callback, lr_monitor], 
        check_val_every_n_epoch=check_interval, 
        default_root_dir=cfg.work_dir,
        accelerator="auto",
        logger=tb_logger)
        
    trainer.fit(model, data_module)

    delta = datetime.now() - start_time
    # time difference in seconds
    print(f"Training time is {delta.total_seconds()/60} minutes")
    
    # print('Best model path:', checkpoint_callback.best_model_path)
    
    print('Evaluated on Last model:')
    result = trainer.test(model, datamodule=data_module)
    
    os.rename(os.path.join(cfg.work_dir, 'last.ckpt'),
            os.path.join(cfg.work_dir, f"seed_{cfg.seed}_acc={str(round(100*result[0]['test_acc'], 2))}_last.ckpt"))
    
    with open(os.path.join(cfg.work_dir, 'results.txt'), "a") as f:
        f.write(cfg.work_dir.split('/')[-1]+'\t'+str(cfg.seed)+'\t'+str(round(100*result[0]['test_acc'], 2))+'\n')
    
    # print('best model')
    # trainer.test(datamodule=data_module)
    

if __name__ == "__main__":
    from mmcv import DictAction
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./cfg/HAM10000/HAM10000_allshot_fac.py', help='path to config file')
    parser.add_argument('--work-dir',
                        help='work directory to save the ckpt and config file')
    parser.add_argument('--func', default='our_main', help='which task to run')
    parser.add_argument('--test',
                        action='store_true',
                        help='whether to enable test mode')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='overwrite parameters in cfg from commandline')
    args = parser.parse_args()
    if not args.test:
        cfg = utils.pre_exp(args.cfg, args.work_dir)
    else:
        from mmcv import Config
        cfg = Config.fromfile(args.cfg)
    cfg.test = args.test
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    main = eval(args.func)
    main(cfg)