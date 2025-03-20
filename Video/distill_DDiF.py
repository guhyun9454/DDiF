import os
import argparse
import numpy as np
import torch
from tqdm import trange
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import shutil
from hyper_params import load_default
from DDiF import DDiF
from utils import set_seed, save_and_print, get_videos, evaluate_synset_nf

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(args.startIt, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    args.channel, args.im_size, args.num_classes, args.mean, args.std = channel, im_size, num_classes, mean, std
    if args.preload:
        save_and_print(args.log_path, "Preloading dataset")
        video_all = []
        label_all = []
        for i in trange(len(dst_train)):
            _ = dst_train[i]
            video_all.append(_[0])
            label_all.append(_[1])
        video_all = torch.stack(video_all)
        label_all = torch.tensor(label_all)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    args.distributed = torch.cuda.device_count() > 1

    save_and_print(args.log_path, "=" * 50)
    save_and_print(args.log_path, f'Hyper-parameters: {args.__dict__}')

    ''' organize the real dataset '''
    indices_class = [[] for c in range(num_classes)]
    save_and_print(args.log_path, "BUILDING DATASET")
    for i, lab in enumerate(label_all):
        indices_class[lab].append(i)

    ''' initialize the synthetic data '''
    synset = DDiF(args)
    synset.init(video_all, label_all, indices_class)

    syn_lr = torch.tensor(0.01)

    ''' training '''
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    save_and_print(args.log_path, '%s training begins' % get_time())

    for it in range(0, args.Iteration + 1):
        save_this_it = False

        ''' Evaluate synthetic data '''
        if it in eval_it_pool and it > 0:
            for model_eval in model_eval_pool:
                save_and_print(args.log_path, '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (args.model, model_eval, it))

                accs_test = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model

                    args.lr_net = syn_lr.detach()
                    _, _, acc_test = evaluate_synset_nf(it_eval, net_eval, synset, testloader, args, mode='none')

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

            ''' visualize and save '''
            save_name = os.path.join(f"{args.save_path}/imgs", "syn_{}.png".format(it))
            video_save, label_save = synset.get(need_copy=True)
            vis_shape = video_save.shape
            video_save = video_save.view(vis_shape[0] * vis_shape[1], vis_shape[2], vis_shape[3], vis_shape[4])
            for ch in range(3):
                video_save[:, ch] = video_save[:, ch] * std[ch] + mean[ch]
            video_save = torch.clamp(video_save, 0, 1)
            save_image(video_save, save_name, nrow=vis_shape[1])
            del video_save, label_save

            if save_this_it:
                synset.save(name=f"DDiF_DM_{args.ipc}ipc#synset_best.pt")

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False

        embed = net.module.embed if args.distributed else net.embed

        ''' update synthetic data '''
        loss = torch.tensor(0.0).to(args.device)
        for c in range(0, num_classes):
            loss_c = torch.tensor(0.0).to(args.device)

            vid_real = get_videos(video_all, indices_class, c, args.batch_real)
            vid_real = vid_real.to(args.device)

            if args.batch_syn > 0:
                indices = np.random.permutation(range(c * synset.num_per_class, (c + 1) * synset.num_per_class))[:args.batch_syn]
            else:
                indices = range(c * synset.num_per_class, (c + 1) * synset.num_per_class)

            vid_syn, lab_syn = synset.get(indices=indices)

            output_real = embed(vid_real).detach()
            output_syn = embed(vid_syn)

            loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

            synset.optim_zero_grad()
            loss_c.backward()
            synset.optim_step()
            loss += loss_c

        loss_avg = loss.item()

        loss_avg /= (num_classes)

        if it % 10 == 0:
            save_and_print(args.log_path, '%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=500, help='epochs to train a model with synthetic data')

    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')

    parser.add_argument('--preload', action='store_true', help="preload all data into RAM")
    parser.add_argument('--frames', type=int, default=16, help='number of frames')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--startIt', type=int, default=0, help='start iteration')
    parser.add_argument('--train_lr', action='store_true', help='train the learning rate')

    ### Basic ###
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sh_file', type=str)
    parser.add_argument('--FLAG', type=str, default="")
    parser.add_argument('--save_path', type=str, default="./results")

    parser.add_argument('--batch_syn', type=int)
    parser.add_argument('--dipc', type=int, default=0)

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

    sub_save_path_1 = f"{args.dataset}_{args.model}_{args.ipc}ipc_{args.dipc}dipc"
    sub_save_path_2 = f"{args.batch_syn}_({args.dim_in},{args.num_layers},{args.layer_size},{args.dim_out})_({args.w0_initial},{args.w0})_({args.epochs_init},{args.lr_nf_init:.0e})_{args.lr_nf:.0e}"

    args.save_path = f"{args.save_path}/{sub_save_path_1}/{sub_save_path_2}#{args.FLAG}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(f"{args.save_path}/imgs")

    shutil.copy(f"./scripts/{args.sh_file}", f"{args.save_path}/{args.sh_file}")
    args.log_path = f"{args.save_path}/log.txt"

    main(args)