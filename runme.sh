#!/bin/bash
python train.py train/M1 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet50
python train.py train/M2 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet152
python train.py train/M3 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 2 --learning_rate 0.001 --vocab_threshold 5 --net resnet152
python train.py train/M4 --num_epochs 5 --batch_size 128 --embed_size 512 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet152
python train.py train/M5 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 1024 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet152
python train.py train/M6 --num_epochs 5 --batch_size 256 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet152
python train.py train/M7 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 10 --net resnet152
python train.py train/M8 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.01 --vocab_threshold 5 --net resnet152
python train.py train/M9 --num_epochs 5 --batch_size 128 --embed_size 128 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet152
python train.py train/M0 --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 256 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5 --net resnet152

