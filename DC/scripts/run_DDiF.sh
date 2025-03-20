cd ..
cuda_id=0
dst="ImageNet"
subset="nette"
res=128
net="ConvNetD5"
ipc=1
sh_file="run_DDiF.sh"
eval_mode="S"
data_path="../data"
save_path="./results"

batch_syn=0 ### 0 means no sampling (use entire synthetic dataset)
dipc=0 ### 0 means utilizing entire allowed budget
lr_nf=5e-5

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DC.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} \
--batch_syn ${batch_syn} \
--dipc ${dipc} \
--lr_nf ${lr_nf} # --zca