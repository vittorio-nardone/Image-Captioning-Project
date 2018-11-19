def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from data_loader_val import get_loader_val
from model import EncoderCNN, EncoderCNN152, DecoderRNN

import math
import torch.utils.data as data
import numpy as np
import os
import argparse

from nltk.translate.bleu_score import sentence_bleu
from time import time, gmtime, strftime
        
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

def get_avg_bleu_score(outputs, references, idx2word):
    score = 0
    for i in range(len(outputs)):
        output = clean_sentence(outputs[i], idx2word)
        reference = clean_sentence(references[i], idx2word)
        score += sentence_bleu([reference], output)
    score /= len(outputs)
    return score
       

def main(args):
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    log_file = os.path.join(args.output_path, 'training_log.txt')       # name of file with saved training loss and perplexity

    # Open the training log file.
    f = open(log_file, 'w')
    f.write(str(args) + '\n')
    f.flush()        
    
    #image transform below.
    transform_train = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])

    #image transform below.
    transform_val = transforms.Compose([ 
        transforms.Resize(256),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
    
    
    # Build data loader.
    data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=args.batch_size,
                         vocab_threshold=args.vocab_threshold,
                         vocab_from_file=False)

    data_loader_val = get_loader_val(transform=transform_val,
                         batch_size=args.batch_size)
    
    
    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder. 
    if (args.net == 'resnet50'):
        encoder = EncoderCNN(args.embed_size)
    elif (args.net == 'resnet152'):
        encoder = EncoderCNN152(args.embed_size)
        
    decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size, num_layers= args.num_layers)

    # Move models to GPU if CUDA is available. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # Define the loss function. 
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # TODO #3: Specify the learnable parameters of the model.
    if (args.net == 'resnet50'):
        params = list(decoder.parameters()) + list(encoder.embed.parameters())
    elif (args.net == 'resnet152'):
        params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

    # TODO #4: Define the optimizer.
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Set the total number of training steps per epoch.
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)
    
    total_step_val = math.ceil(len(data_loader_val.dataset.caption_lengths) / data_loader_val.batch_sampler.batch_size)

    start_time = time()
    
    epoch_stats = np.zeros((args.num_epochs, 5))
    
    for epoch in range(1, args.num_epochs+1): 
    
        encoder.train()
        decoder.train()
        
        epoch_time = time()
        epoch_loss = 0
    
        for i_step in range(1, total_step+1):
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler
        
            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)
        
            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()
        
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)
        
            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            epoch_loss += loss.item()
        
            # Backward pass.
            loss.backward()
        
            # Update the parameters in the optimizer.
            optimizer.step()
            
            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, args.num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        
            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()
        
            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()
        
            # Print training statistics (on different line).
            if i_step % args.log_step == 0:
                print('\r' + stats)
                # If debug option is enable, exit soon  
                if (args.debug == True):
                    break

        epoch_stats[epoch-1,0] = epoch_loss / total_step    
        epoch_stats[epoch-1,1] = time() - epoch_time
                
        encoder.eval()
        decoder.eval()     
        
        epoch_time = time()
        epoch_loss = 0 
        epoch_bleu_score = 0
                
        for i_step in range(1, total_step_val+1):
           
            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader_val.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader_val.batch_sampler.sampler = new_sampler
        
            # Obtain the batch.
            images, captions = next(iter(data_loader_val))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)
            
        
            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)
        
            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            epoch_loss += loss.item()
        
            # Get predictions
            outputs = decoder.sample(features.unsqueeze(1)) 
            epoch_bleu_score += get_avg_bleu_score(outputs, captions.tolist(), data_loader_val.dataset.vocab.idx2word)
            
        
            # Get validation statistics.
            stats = 'Validation Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Bleu: %5.4f' % (epoch, args.num_epochs, i_step, total_step_val, loss.item(), epoch_bleu_score/i_step)
        
            # Print validation statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()
        
            # Print validation statistics to file.
            if i_step == total_step_val: 
                f.write(stats + '\n')
                f.flush()
                
            # If debug option is enabled, exit soon    
            if i_step % args.log_step == 0:
                if (args.debug == True):
                    f.write(stats + '\n')
                    f.flush()                    
                    break                
        
        print("\n")    

        epoch_stats[epoch-1,2] = epoch_loss / total_step_val    
        epoch_stats[epoch-1,3] = time() - epoch_time
        epoch_stats[epoch-1,4] = epoch_bleu_score / total_step_val    
        
        
        # Save the weights.
        if epoch % args.save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(args.output_path, 'decoder-%d.pkl' % epoch))
            torch.save(encoder.state_dict(), os.path.join(args.output_path, 'encoder-%d.pkl' % epoch))
        

    tot_time = time() - start_time
    elapsed = "\n** Total Elapsed Runtime:" + strftime("%H:%M:%S", gmtime(tot_time))
    print(elapsed)
    f.write(elapsed + '\n')
    f.flush()    
            
    # Close the training log file.
    f.close()
    
    np.savetxt(os.path.join(args.output_path,"stats.csv"), epoch_stats, delimiter=",")    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, nargs=1, help='path for saving trained model and output files')
    parser.add_argument('--vocab_threshold', type=int, default=5, help='minimum word count threshold (default 5)')
    parser.add_argument('--log_step', type=int , default=100, help='step size for printing log info (default 100)')
    parser.add_argument('--save_step', type=int , default=1, help='save trained models every N epoch (default 1)')
    
    # Model parameters
    parser.add_argument('--net', default='resnet50', const='resnet50', nargs='?', choices=['resnet50', 'resnet152'],
                    help='encoder pretrained network (default resnet50")')    
    
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors (default 256)')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states (default 512)')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm (default 1)')
    
    parser.add_argument('--num_epochs', type=int, default=5, help='training epochs (default 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size (default 128)')
    #parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001, help='training learning rate (default 0.001)')
    parser.add_argument('--debug', action='store_true', help='enable debug mode (one batch train & validation)')
    args = parser.parse_args()
    
    args.output_path = args.output_path[0]
    
    # Debug options
    if (args.debug == True):
        args.num_epochs = 1
        args.log_step = 10
    
    print(args)
    main(args)
