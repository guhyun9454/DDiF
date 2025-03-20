cd ..
cuda_id=0
dst="ModelNet"
res=32
net="Conv3DNet"
ipc=1
sh_file="run_DDiF#DC.sh"
eval_mode="S"
data_path="../data"
save_path="./results"

batch_syn=0 ### 0 means no sampling (use entire synthetic dataset)
dipc=0 ### 0 means utilizing entire allowed budget
lr_nf=1e-4

CUDA_VISIBLE_DEVICES=${cuda_id} python main_DC.py \
--dataset ${dst} --res ${res} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} \
--batch_syn ${batch_syn} \
--dipc ${dipc} \
--lr_nf ${lr_nf}