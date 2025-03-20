cd ..
cuda_id=0
dst="miniUCF101"
net="ConvNet3D"
ipc=1
sh_file="run_DDiF.sh"
eval_mode="SS"
data_path="../data"
save_path="./results"

batch_syn=0 ### 0 means no sampling (use entire synthetic dataset)
dipc=1 ### 0 means utilizing entire allowed budget
lr_nf=1e-4

CUDA_VISIBLE_DEVICES=${cuda_id} python distill_DDiF.py \
--dataset ${dst} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} \
--batch_syn ${batch_syn} \
--dipc ${dipc} \
--lr_nf ${lr_nf} \
--preload