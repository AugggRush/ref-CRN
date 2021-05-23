import random
import torch
import os
import json
from tqdm import tqdm
from dataset import LJspeechDataset, collate_fn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.crnn import CRNN


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = CRNN(**CRNN_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(output_directory, epochs, learning_rate, iters_per_checkpoint, batch_size, seed, checkpoint_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    criterion = torch.nn.L1Loss()
    model = CRNN(**CRNN_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    torch.autograd.set_detect_anomaly(True)
    # load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                            optimizer)
    
        iteration += 1

    trainset = LJspeechDataset(**data_config)
    # my_collate = collate_fn(trainset)
    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,\
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                pin_memory=False,
                                drop_last=True)
    
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory)

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))

    for epoch in range(epoch_offset, epochs):
        epoch_ave_loss = 0
        print("Epoch: {}".format(epoch))
        for i, batch in tqdm(enumerate(train_loader)):
            model.zero_grad()

            # zeroPadded_batch = pad_sequence(batch, batch_first=True)

            netFeed = batch[:, :-1, :]
            netTarget = batch[:, 1:, :]
            netTarget = torch.autograd.Variable(netTarget.cuda())
            netFeed = torch.autograd.Variable(netFeed.cuda())

            netOutput = model(netFeed)

            loss = criterion(netOutput, netTarget)

            reduced_loss = loss.item()

            loss.backward()

            optimizer.step()

            if (iteration % iters_per_checkpoint == 0):
                print("{}:\t{:.9f}".format(iteration, reduced_loss))
            iteration += 1
            epoch_ave_loss += reduced_loss

        checkpoint_path = "{}/CRNN_net_{}".format(
            output_directory, epoch)
        save_checkpoint(model, optimizer, learning_rate, iteration,
                        checkpoint_path)
        epoch_ave_loss = epoch_ave_loss / i
        print("Epoch: {}, the average epoch loss: {}".format(epoch, epoch_ave_loss))


if  __name__ == "__main__":

    config = 'config.json'
    with open(config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global CRNN_config
    CRNN_config = config["CRNN_config"]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(**train_config)