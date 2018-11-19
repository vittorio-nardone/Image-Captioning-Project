import os
from data_loader import CoCoDataset

import torch
import torch.utils.data as data

def get_loader_val(transform,
               batch_size=1,
               vocab_file='./vocab.pkl',
               num_workers=0,
               cocoapi_loc='/opt'):
    
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')
    
    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode='train',
                          batch_size=batch_size,
                          vocab_threshold=None,
                          vocab_file=vocab_file,
                          start_word="<start>",
                          end_word="<end>",
                          unk_word="<unk>",
                          annotations_file=annotations_file,
                          vocab_from_file=True,
                          img_folder=img_folder)    

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=dataset.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    return data_loader

