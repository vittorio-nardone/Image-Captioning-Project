def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
import pickle 
import os
from torchvision import transforms 
from model import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def clean_sentence(output, idx2word):
    sentence = ''
    for x in output:
        word = idx2word[x] 
        if word == '<end>':
            break
        elif word == '<start>':
            pass
        elif word == '.':
            sentence += word 
        else:
            sentence += ' ' + word 
    return sentence.strip()

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    if (args.net == 'resnet50'):
        encoder = EncoderCNN(args.embed_size).eval()
    elif (args.net == 'resnet152'):
        encoder = EncoderCNN152(args.embed_size).eval()    
    
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    
    # Convert word_ids to words
    sentence = clean_sentence(sampled_ids, vocab.idx2word)
    
    # Print out generated caption
    print("File: '{}' - Caption: '{}'".format(args.image, sentence))
    
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-10.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-10.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--net', default='resnet50', const='resnet50', nargs='?', choices=['resnet50', 'resnet152'],
                    help='encoder pretrained network (default resnet50")')        
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    args = parser.parse_args()
    main(args)
