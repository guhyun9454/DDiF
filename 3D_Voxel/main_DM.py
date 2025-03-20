import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import torch
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, set_seed, save_and_print, get_voxels
import time

import shutil
from hyper_params import load_default
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from DDiF import DDiF
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM')
    parser.add_argument('--dataset', type=str, default='ModelNet')
    parser.add_argument('--model', type=str, default='Conv3DNet')
    parser.add_argument('--ipc', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='S')
    parser.add_argument('--num_exp', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=5)
    parser.add_argument('--epoch_eval_train', type=int, default=1000)
    parser.add_argument('--Iteration', type=int, default=20000)
    parser.add_argument('--lr_net', type=float, default=0.01)
    parser.add_argument('--batch_real', type=int, default=256)
    parser.add_argument('--batch_train', type=int, default=256)
    parser.add_argument('--dsa_strategy', type=str, default="")
    parser.add_argument('--data_path', type=str, default='../data')

    ### Basic ###
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--batch_syn', type=int)
    parser.add_argument('--dipc', type=int, default=0)
    parser.add_argument('--res', type=int)

    ### DDiF ###
    parser.add_argument('--dim_in', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--layer_size', type=int)
    parser.add_argument('--dim_out', type=int)
    parser.add_argument('--w0_initial', type=float)
    parser.add_argument('--w0', type=float)
    parser.add_argument('--lr_nf', type=float)
    parser.add_argument('--epochs_init', type=int, default=5000)
    parser.add_argument('--lr_nf_init', type=float, default=5e-4)

    args = parser.parse_args()
    set_seed(args.seed)
    args = load_default(args)

    # args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if args.dataset == "ModelNet":
        args.data_path += "/modelnet40_normal_resampled"
    elif args.dataset == "ShapeNet":
        args.data_path += "/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    else:
        save_and_print(args.log_path, f"Invalid dataset: {args.dataset}")
        exit()

    sub_save_path_1 = f"{args.dataset}_{args.res}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"
    sub_save_path_2 = f"DM#{args.batch_syn}_({args.dim_in},{args.num_layers},{args.layer_size},{args.dim_out})_({args.w0_initial},{args.w0})_({args.epochs_init},{args.lr_nf_init:.0e})_{args.lr_nf:.0e}"

    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        # os.makedirs(f"{args.save_path}/imgs")

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, resolution=args.res)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    for exp in range(args.num_exp):
        save_and_print(args.log_path, f'\n================== Exp {exp} ==================\n ')
        save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

        ''' organize the real dataset '''
        voxels_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        save_and_print(args.log_path, "BUILDING DATASET")
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            voxels_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(sample[1])

        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)
        voxels_all = torch.cat(voxels_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        ''' initialize the synthetic data '''
        synset = DDiF(args)
        synset.init(voxels_all, labels_all, indices_class)

        ''' training '''
        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        save_and_print(args.log_path, '%s training begins' % get_time())

        for it in range(args.Iteration+1):
            save_this_it = False

            ''' Evaluate synthetic data '''
            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))

                    save_and_print(args.log_path, f'DSA augmentation strategy: {args.dsa_strategy}')
                    save_and_print(args.log_path, f'DSA augmentation parameters: {args.dsa_param.__dict__}')

                    accs_test = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        voxel_syn_eval, label_syn_eval = synset.get(need_copy=True)
                        _, _, acc_test = evaluate_synset(it_eval, net_eval, voxel_syn_eval, label_syn_eval, testloader, args)
                        accs_test.append(acc_test)
                    accs_test = np.array(accs_test)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                        torch.save({"best_acc": best_acc, "best_std": best_std}, f"{args.save_path}/best_performance.pt")
                    save_and_print(args.log_path, 'Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    save_and_print(args.log_path, f"{args.save_path}")
                    save_and_print(args.log_path, f"{it:5d} | Accuracy/{model_eval}: {acc_test_mean}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Accuracy/{model_eval}: {best_acc[model_eval]}")
                    save_and_print(args.log_path, f"{it:5d} | Std/{model_eval}: {acc_test_std}")
                    save_and_print(args.log_path, f"{it:5d} | Max_Std/{model_eval}: {best_std[model_eval]}")
                    del voxel_syn_eval, label_syn_eval

                if save_this_it:
                    synset.save(name=f"DDiF_DM_{args.ipc}ipc#synset_best.pt")

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                loss_c = torch.tensor(0.0).to(args.device)

                vox_real = get_voxels(voxels_all, indices_class, c, args.batch_real)

                if args.batch_syn > 0:
                    indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
                else:
                    indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

                vox_syn, lab_syn = synset.get(indices=indices)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    vox_real = DiffAugment(vox_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    vox_syn = DiffAugment(vox_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = embed(vox_real).detach()
                output_syn = embed(vox_syn)

                loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                synset.optim_zero_grad()
                loss_c.backward()
                synset.optim_step()
                loss += loss_c

            loss_avg = loss.item()

            loss_avg /= (num_classes)

            if it%10 == 0:
                save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

if __name__ == '__main__':
    main()


