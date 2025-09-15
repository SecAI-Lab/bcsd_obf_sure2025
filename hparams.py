# Optimizer params
lr = 0.0005
adam_beta1 = 0.9        # [0.0-1.0]
adam_beta2 = 0.999      # [0.0-1.0]
adam_weight_decay_rate = 0.01
epsilon = 1e-6          # [> 0.0]
mlm_clip_grad_norm = 1.0
clip_grad_norm = True
warmup = 0.1

# Trainer params
pretrain_epochs = 15
finetune_epochs = 10
log_freq = 1000
save_train_loss = 2000
save_model = 10000
save_checkpoint = 10000
save_runs = 500
pretrain_batch_size = 224
num_hidden = 128


finetune_batch_size = 128
vocab_size = 10000
hidden_size = 128
hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
num_hidden_layers = 8
num_attention_heads = 8
enc_maxlen = 512    
save_steps = 1000               
use_fp16 = True                
do_whole_word_mask = True       
mlm_prob = 0.15                
max_grad_norm = 1.0