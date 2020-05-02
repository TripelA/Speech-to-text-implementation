import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
from warpctc_pytorch import CTCLoss
# from logger import VisdomLogger, TensorBoardLogger


from data_loader_stripped import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
# from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler

from decoder_stripped import GreedyDecoder
from model import DeepSpeech, supported_rnns
from test_stripped import evaluate
from utils_stripped import reduce_tensor, check_loss, remove_parallel_wrapper

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.cpu().numpy()


class DefaultArguments():
    """ Use this class to change values of model, data and training parameters"""

    def __init__(self):
        self.id = 'Deepspeech training'

        # TODO check paths
        self.train_manifest = 'data/voxforge_train_manifest_v2.csv'
        self.val_manifest = 'data/voxforge_test_manifest.csv'

        # sampling_rate = 16k says that this audio was recorded(sampled) with a sampling frequency of 16k. In other
        # words, while recording this file we were capturing 16000 amplitudes every second.
        self.sample_rate = 16000

        self.batch_size = 20
        self.num_workers = 0
        self.labels_path = 'labels.json'
        self.window_size = .02  # 'Window size for spectrogram in seconds'
        self.window_stride = .01  # 'Window stride for spectrogram in seconds'
        self.window = 'hamming'  # 'Window type for spectrogram generation'
        self.hidden_size = 1000  # 'Hidden size of RNNs'
        self.hidden_layers = 5  # 'Number of RNN layers'
        self.rnn_type = 'lstm'  # 'Type of the RNN. rnn|gru|lstm are supported'
        self.epochs = 30  # Number of training epochs
        self.cuda = 'cuda'  # Use cuda to train model'
        self.lr = 3e-4  # 'initial learning rate'
        self.momentum = 0.9  # 'momentum'
        self.max_norm = 400  # 'Norm cutoff to prevent explosion of gradients'
        self.learning_anneal = 1.1  # 'Annealing applied to learning rate every epoch'
        self.silent = False  # 'Turn off progress tracking per iteration'
        self.checkpoint = False  # 'Enables checkpoint saving of model'
        self.checkpoint_per_batch = 0  # Save checkpoint per batch. 0 means never save'
        self.visdom = False  # Turn on visdom graphing'
        self.tensorboard = False  # 'Turn on tensorboard graphing'
        self.log_dir = 'visualize/deepspeech_final'  # 'Location of tensorboard log'
        self.log_params = False  # 'Log parameter values and gradients'
        self.id = 'Deepspeech training'  # 'Identifier for visdom/tensorboard run'
        self.save_folder = 'models/'  # 'Location to save epoch models'
        self.model_path = 'models/iteration5.pth'  # 'Location to save best validation model'
        # TODO check path
        self.continue_from = 'librispeech_pretrained_v2.pth'  # continue from checkpoint model
        self.finetune = True  # 'Finetune the model from checkpoint "continue_from"'
        self.speed_volume_perturb = False  # 'Use random tempo and gain perturbations.'
        self.spec_augment = False
        self.noise_dir = None  # 'Directory to inject noise into audio. If default, noise Inject not added'
        self.noise_prob = 0.4  # 'Probability of noise being added per sample'
        self.noise_min = 0.0  # 'Minimum noise level to sample from. (1.0 means all noise, not original signal)'
        self.noise_max = 0.5  # 'Maximum noise levels to sample from. Maximum 1.0'
        self.no_shuffle = False  # 'Turn off shuffling and sample from dataset based on sequence length (smallest to
        # largest)'
        self.no_sortaGrad = False  # 'Turn off ordering of dataset on sequence length for the first epoch.'
        self.bidirectional = True  # 'Turn off bi-directional RNNs, introduces lookahead convolution'
        self.dist_url = 'tcp://127.0.0.1:1550'  # 'url used to set up distributed training'
        self.dist_backend = 'nccl'  # distributed backend
        self.world_size = 1  # 'number of distributed processes'
        self.rank = 0  # 'The rank of this process'
        self.gpu_rank = None  # 'If using distributed parallel for multi-gpu, sets the GPU for the process'
        self.seed = 123456  # Seed to generators
        self.opt_level = 'O1'
        self.keep_batchnorm_fp32 = None
        self.loss_scale = 1  # Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of
        # gradients'
        self.distributed = False
        self.no_sorta_grad = False


class AverageMeter(object):
    """Computes and stores the average and current value, used for evaluation and epoch time computation and ctc loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':

    # load the default arguments
    args = DefaultArguments()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set device to cuda
    device = torch.device("cuda" if args.cuda else "cpu")
    os.system("export CUDA_VISIBLE_DEVICES=1")

    # if the number of distributed process is 1, set the value to True
    args.distributed = args.world_size > 1

    main_proc = True
    # device = torch.device("cuda" if args.cuda else "cpu")

    # if we want to use distributed programming
    # if args.distributed:
    #     if args.gpu_rank:
    #         torch.cuda.set_device(int(args.gpu_rank))
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)
    #     main_proc = args.rank == 0  # Only the first proc should save models
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists

    # set up the variables for the number of epochs
    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None

    # visualization tool for check progress of model training
    # if main_proc and args.visdom:
    #     visdom_logger = VisdomLogger(args.id, args.epochs)
    # if main_proc and args.tensorboard:
    #     tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    avg_loss, start_epoch, start_iter, optim_state, amp_state = 0, 0, 0, None, None

    # start from the pretrained models
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)

        # Load all tensors onto the CPU, using a function ( refer to torch.serialization doc to know more)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)

        # load pretrained model
        model = DeepSpeech.load_model_package(package)

        # set labels A-Z, -, ' ', total 29
        labels = model.labels

        audio_conf = model.audio_conf

        if not args.finetune:  # Don't want to restart training
            optim_state = package['optim_dict']
            amp_state = package['amp']
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1

            # get what was the last avg loss
            avg_loss = int(package.get('avg_loss', 0))

            # get evaluation metrics for ctc loss, wer, cer
            loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], \
                                                     package['wer_results']
            best_wer = wer_results[start_epoch]
            # if main_proc and args.visdom:  # Add previous scores to visdom graph
            #     visdom_logger.load_previous_values(start_epoch, package)
            # if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
            #     tensorboard_logger.load_previous_values(start_epoch, package)
    # train new model
    else:
        # read labels
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        # create audio configuration dictionary
        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        # rnn type either GRU or LSTM
        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

        # create network architecture
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)

    # choose the algorithm to decode the model output
    decoder = GreedyDecoder(labels)

    # read the train dataset
    # representation of frequencies of a given signal with time is called a spectrogram
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.speed_volume_perturb,
                                       spec_augment=args.spec_augment)

    # read the test dataset
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    # sample the train sampler depending on the batchsize
    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        # if we are using distributed programing on multiple GPUs
        train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                    num_replicas=args.world_size, rank=args.rank)

    # data generator for train and test
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    # shuffle the batches after every epoch to improve the performance
    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    parameters = model.parameters()

    # Declare model and optimizer as usual, with default (FP32) precision
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    # amp is automatic mixed precision
    # Allow Amp to perform casts as required by the opt_level
    # Amp allows users to easily experiment with different pure and mixed precision modes.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

    # load optimizer state
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    # load precision state
    if amp_state is not None:
        amp.load_state_dict(amp_state)

    if args.distributed:
        model = DistributedDataParallel(model)
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # create class objects
    criterion = CTCLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # start the training epoch
    for epoch in range(start_epoch, args.epochs):

        model.train()
        end = time.time()
        start_epoch_time = time.time()

        # load data using generator audio data loader
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break

            # input_percentages = sample seq len/ max seq len in the batch
            # target sizes = len of target in every seq
            inputs, targets, input_percentages, target_sizes = data

            # every input size input % * max seq length size(3)
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)

            # model outputs batch * max seq length (T) * 29(labels)
            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss

            # calculate ctc loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            # if distributed gather ctc loss
            if args.distributed:
                loss = loss.to(device)
                loss_value = reduce_tensor(loss, args.world_size).item()
            else:
                loss_value = loss.item()

            # Check to ensure valid loss was calculated, there is no inf or nan
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()
                # compute gradient

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            # add epoch loss
            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print the output on the console
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
            # if you want to save output after every batch, default set to 0
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0 and main_proc:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(remove_parallel_wrapper(model),
                                                optimizer=optimizer,
                                                amp=amp,
                                                epoch=epoch,
                                                iteration=i,
                                                loss_results=loss_results,
                                                wer_results=wer_results,
                                                cer_results=cer_results,
                                                avg_loss=avg_loss),
                           file_path)
            del loss, out, float_out

        # average loss across all batches
        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        # evalulate results on test dataset
        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=decoder,
                                             target_decoder=decoder)
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        values = {
            'loss_results': loss_results,
            'cer_results': cer_results,
            'wer_results': wer_results
        }
        # if args.visdom and main_proc:
        #     visdom_logger.update(epoch, values)
        # if args.tensorboard and main_proc:
        #     tensorboard_logger.update(epoch, values, model.named_parameters())
        #     values = {
        #         'Avg Train Loss': avg_loss,
        #         'Avg WER': wer,
        #         'Avg CER': cer
        #     }

        # if you have to save file after every epoch
        if main_proc and args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(remove_parallel_wrapper(model),
                                            optimizer=optimizer,
                                            amp=amp,
                                            epoch=epoch,
                                            loss_results=loss_results,
                                            wer_results=wer_results,
                                            cer_results=cer_results),
                       file_path)
        # anneal lr Learning rate annealing is reducing the rate after every epoch in order to not miss the local
        # minimum and avoid oscillation
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        # if the best model is found than previous iteration, overwrite the model with better model
        if main_proc and (best_wer is None or best_wer > wer):
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(remove_parallel_wrapper(model),
                                            optimizer=optimizer,
                                            amp=amp, epoch=epoch,
                                            loss_results=loss_results,
                                            wer_results=wer_results,
                                            cer_results=cer_results)
                       , args.model_path)
            best_wer = wer
            avg_loss = 0

        # if you want to shuffle argument after every epoch
        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
