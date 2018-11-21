import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class EncoderCNN152(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN152, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        # Added an extra Batch Normalization layer
        features = self.bn(self.embed(features))
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        
        # Embed captions (except last one, because it's not submitted to model)
        embeddings = self.embed(captions[:,:-1]) 
        # Add images features on top of captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # Submit input to LSTM
        hiddens, _ = self.lstm(embeddings)
        # Submit LSTM output to Dense layer 
        outputs = self.linear(hiddens)
        return outputs        

    def sample(self, inputs, states=None, max_len=20, validation=False):
        """Generate captions for given image(s) features."""
        # Check if single/batch. If single image is submitted, add dummy batch dim.
        if (len(inputs.shape) == 2):   
            inputs = inputs.unsqueeze(1)
        
        # Predictions
        sampled_ids = []
        
        # In case of validation, prepare tensor to store model output.
        if (validation == True):
            outputs_val = torch.zeros([inputs.shape[0], max_len, self.vocab_size], dtype=torch.float32)
        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            
            if (validation == True):                             # in case of validation, save model output
                outputs_val[:,i,:] = outputs                     
                
            predicted = outputs.argmax(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)                        
            
            #prepare input to be used in next iteration
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        
        sampled_ids = torch.stack(sampled_ids, 1).tolist()       # sampled_ids: (batch_size, max_seq_length)
        
        # check if single/batch. If single image is submitted, remove dummy batch dim
        if (inputs.shape[0] == 1):                              
            sampled_ids = sampled_ids[0]                        
            if (validation == True):               
                 outputs_val = outputs_val[0]
        
        # in case of validation, return both predictions and outputs
        if (validation == True):
            return_val = (sampled_ids, outputs_val)
        else: # return predictions only
            return_val = sampled_ids
        
        return return_val
    
