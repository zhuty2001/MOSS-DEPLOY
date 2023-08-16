# Moss Train
根据moss_deploy中的操作说明，已成功将moss搭建到本地，本篇将解释说明如何基于已有的moss网络框架，对网络进行训练。

## 环境配置

为进行moss网络的训练，需要安装一定的依赖库，首先确保环境变量中的cuda为正确的cuda，使用以下指令指定cuda位置，对应地址为本地cuda的安装地址。

```bash
export CUDA_HOME='usr/local/cuda-12.2' 
```

运行以下命令安装deepspeed框架

```bash
pip3 install deepspeed 
```

运行以下命令安装accelerate

```bash
pip install accelerate 
```

安装完成后，使用以下命令accelerate在本地的参数

```bash
accelerate config 
```

本文的本地服务器对应的配置如下，根据需求自行调整参数配置

```bash
(base) winner@winner-PR4904P:~$ accelerate config
[2023-08-16 16:19:59,988] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                                                                                                                                                                                
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                                                                                                                        
multi-GPU                                                                                                                                                                                                                                                   
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                                                                                                                                  
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                                                                                                                                           
Do you want to use DeepSpeed? [yes/NO]: yes                                                                                                                                                                                                                 
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO                                                                                                                                                                                      
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?                                                                                                                                                                                                    
3                                                                                                                                                                                                                                                           
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?                                                                                                                                                                                                                          
cpu                                                                                                                                                                                                                                                         
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?                                                                                                                                                                                                                                
cpu                                                                                                                                                                                                                                                         
How many gradient accumulation steps you're passing in your script? [1]: 1                                                                                                                                                                                  
Do you want to use gradient clipping? [yes/NO]: NO                                                                                                                                                                                                          
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: NO                                                                                                                                                                              
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: NO
How many GPU(s) should be used for distributed training? [1]:4
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp16                                                                                                                                                                                                                                                        
accelerate configuration saved at /home/winner/.cache/huggingface/accelerate/default_config.yaml  
```

## 修改本地文件

接下来找到`MOSS/configs/sft.yaml`文件,需根据本地配置及需求进行更改

![1692174941645](https://github.com/zhuty2001/moss_deploy/assets/68087747/bc486542-a649-4515-b426-c5a339640d73)

接下来需要准备训练数据集，将对应的训练数据按[conversation_without_plugins](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data/conversations/conversation_without_plugins) 格式处理并放到 `SFT_data` 目录中即可。

## 训练网络

最后，我们在`MOSS`文件夹下创建文件`run.sh`并将以下内容放入

```bash
num_machines=1
num_processes=$((num_machines * 4))
machine_rank=0

accelerate launch \
	--config_file ./configs/sft.yaml \
	--num_processes $num_processes \
	--num_machines $num_machines \
	--machine_rank $machine_rank \
	--deepspeed_multinode_launcher standard finetune_moss.py \
	--model_name_or_path fnlp/moss-moon-003-base \
	--data_dir ./SFT_data \
	--output_dir ./ckpts/moss-moon-003-sft \
	--log_dir ./train_logs/moss-moon-003-sft \
	--n_epochs 2 \
	--train_bsz_per_gpu 4 \
	--eval_bsz_per_gpu 4 \
	--learning_rate 0.000015 \
	--eval_step 200 \
	--save_step 2000
```
其中`num_machines`以及`num_processes`根据本地情况调整，模型地址，数据目录等信息同理。

最后在命令行中运行`run.sh`即可完成对模型的训练。

```bash
bash run.sh
```
