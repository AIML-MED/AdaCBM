proj_name = "HAM10000"
concept_root = 'datasets/HAM10000/concepts/'
img_split_path = 'datasets/HAM10000/splits'
img_path = 'datasets/HAM10000/images'

img_ext = ''
raw_sen_path = concept_root + 'concepts_raw.npy'
concept2cls_path = concept_root + 'concept2cls.npy'
cls_name_path = concept_root + 'cls_names.npy'
num_cls = 7

## data loader
on_gpu = True

# concept select
num_concept = num_cls * 50

# weight matrix fitting
lr = 0.0005
max_epochs = 300

# CLIP Backbone
clip_model = 'ViT-L/14'
n_shots = "all"
data_root = 'exp/HAM10000/'
cls_names = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions', 
            'dermatofibroma', 'melanocytic nevi', 
            'melanoma', 'vascular lesions']
dataset = 'HAM10000'
bs = 256 
num_layers=2
freeze_backbone = True
use_img_norm = False
use_txt_norm = False
use_rand_init = False
init_val = 1.0

# loss weight
seed = 123
pearson_weight=0.9
concept_path = './doctor_concepts/HAM10000.json'



