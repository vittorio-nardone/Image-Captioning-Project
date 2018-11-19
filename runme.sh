#!/bin/bash
python train.py train/A --num_epochs 5 --batch_size 64 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/B --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/C --num_epochs 5 --batch_size 256 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/D --num_epochs 5 --batch_size 512 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/E --num_epochs 5 --batch_size 128 --embed_size 128 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/F --num_epochs 5 --batch_size 128 --embed_size 512 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/G --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 256 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/H --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 1024 --num_layers 1 --learning_rate 0.001 --vocab_threshold 5
python train.py train/I --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 2 --learning_rate 0.001 --vocab_threshold 5
python train.py train/L --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 3 --learning_rate 0.001 --vocab_threshold 5
python train.py train/M --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.1 --vocab_threshold 5
python train.py train/N --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.01 --vocab_threshold 5
python train.py train/O --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.0001 --vocab_threshold 5
python train.py train/P --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 2
python train.py train/Q --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 10
python train.py train/R --num_epochs 5 --batch_size 128 --embed_size 256 --hidden_size 512 --num_layers 1 --learning_rate 0.001 --vocab_threshold 15

