import torch as th
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

def utility_function(pearson_values, cls_selected_feature_indices, selected_indices, num_concepts_per_cls, gamma):
    i = 0
    if gamma>1.0 or gamma<0:
        return selected_indices
    while len(selected_indices)<num_concepts_per_cls:
        t = cls_selected_feature_indices[i]
        selected_indices.append(t)
        R = list(np.argwhere(pearson_values[i,:]>gamma).flatten())
        if num_concepts_per_cls - len(selected_indices) <= len(cls_selected_feature_indices) - len(R):
            cls_selected_feature_indices = np.delete(cls_selected_feature_indices, R)
            idx = list(set(range(pearson_values.shape[0])).difference(R))
            pearson_values = pearson_values[np.ix_(idx,idx)]
        else:
            selected_indices = utility_function(pearson_values, 
                                            cls_selected_feature_indices, 
                                            selected_indices,
                                            num_concepts_per_cls, 
                                            gamma+0.1)
        
    return selected_indices

def select_features(dot_product, y, selected_concept2cls, num_concepts_per_cls=350, alpha=0.9, concept2type=None):
    print('Pearson weight:', alpha)
    num_features = dot_product.shape[1]
    classes = y.unique()
    
    all_t_stats = []
    
    for cls in classes.numpy():
        indices = ((selected_concept2cls == cls).nonzero(as_tuple=True)[0])
        for i in indices:
            v_c = dot_product[y==cls, i].numpy()
            v_c1 = dot_product[y!=cls, i].numpy()
            result = ttest_ind(v_c, v_c1, equal_var=False, alternative='greater')
            all_t_stats.append(result.statistic)
            
    p_values = np.array(all_t_stats)
    
    # selecting num_concept_per_class for each concept 
    selected_features_indices2 = np.array([])
    cls_selected_concept2cls = selected_concept2cls.numpy()
    cls_p_values = np.argsort(p_values)[::-1]

    for cls_id in classes.numpy():
        cls_selected_feature_indices = cls_p_values[cls_selected_concept2cls[cls_p_values]==cls_id]#[:num_concept_per_class]
        features0 = np.array([dot_product[y==cls_id,idx].numpy() for idx in cls_selected_feature_indices])
        R1 = np.corrcoef(features0)
        R1 = np.absolute(R1)
        
        selected_indices = []
        selected_indices = utility_function(R1, cls_selected_feature_indices, selected_indices, 
                        num_concepts_per_cls-len(selected_indices), alpha)
        selected_features_indices2 = np.append(selected_features_indices2,
            np.array(selected_indices)
        )
    
    selected_features_indices2 = np.sort(selected_features_indices2).astype(int)

    return selected_features_indices2, p_values

def our_selection(img_feat, concept_feat, concept2cls, num_concepts, num_images_per_class, 
                 *args, **kwargs):
    assert num_concepts > 0
    
    num_cls = len(num_images_per_class)
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    sorted_concept2cls_idx = np.argsort(concept2cls)
    sorted_concept2cls = th.from_numpy(concept2cls[sorted_concept2cls_idx])
    sorted_concept_feat = concept_feat[th.from_numpy(sorted_concept2cls_idx), :]
    sorted_concept2type, concept2type = None, None
    if 'concept2type' in kwargs:
        concept2type = kwargs['concept2type']
        if concept2type is not None:
            sorted_concept2type = th.from_numpy(concept2type[sorted_concept2cls_idx])
    if 'pearson_weight' in kwargs:
        alpha=float(kwargs['pearson_weight'])
    
    dot_product = img_feat @ sorted_concept_feat.t()
    
    class_start_indices = [0]
    for i in range(num_cls):
        class_start_indices.append(sum(num_images_per_class[:i+1]))

    # Create an array of class labels efficiently using NumPy
    class_labels = np.zeros(dot_product.shape[0], dtype=int)
    for i in range(len(class_start_indices) - 1):
        start, end = class_start_indices[i], class_start_indices[i + 1]
        class_labels[start:end] = i

    # Convert the class labels array into a PyTorch tensor
    class_labels_tensor = th.from_numpy(class_labels)
    selected_idx, _ = select_features(dot_product, class_labels_tensor, 
                            sorted_concept2cls, num_concepts_per_cls, 
                            concept2type=sorted_concept2type,
                            alpha=alpha)
    
    # selected_idx is for sorted_concept2cls 
    return th.from_numpy(sorted_concept2cls_idx[selected_idx]), None



