import torch
from torch import optim
from tqdm import tqdm
from pathlib import Path
from models import get_model
from dataset import get_dataset
from helper import model_text, model_out_to_text
import wandb
import sys


def train(config, log, device):
    Model = get_model(config['model'])
    Dataset = get_dataset(config['dataset'])

    model = Model(Dataset.clasess).to(device)

    wandb.init(
        # set the wandb project where this run will be logged
        project="text_detector_1",
        config=config,
        mode='disabled' if log == False else 'run'
    )

    # dataset
    dataset = Dataset(size=config['train_size'], only_synt=False)
    eval_dataset = Dataset(
        Path('../datasets/detect_val_data'), only_synt=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=8, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'])

    # model test
    # for img, label, lengths in eval_dataloader:
    #     out, loss = model(img, targets=label, lengths=lengths)
    #     print(out.shape, loss)

    # # train setting
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    # train loop
    lrs = []
    losses = []
    model.train()
    for epoch in range(config['epochs']):
        epoch_loss = 0
        batches = 0
        print('epoch -', epoch)
        lrs.append(optimizer.param_groups[0]['lr'])
        print('learning rate', lrs[-1])
        pbar = tqdm(dataloader)
        for img, targets, lengths in pbar:
            targets = targets.to(device)
            img = img.to(device)

            # data check
            # temp_img = img[0].permute([1, 2, 0])
            # cv2.imshow(model_text(targets[0]), temp_img.cpu().detach().numpy())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            lengths = lengths.to(device)
            batches += 1
            optimizer.zero_grad()
            _, loss = model(img.to(device), targets, lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(
                {'loss': epoch_loss/batches})
        batch_loss = epoch_loss/batches
        losses.append(batch_loss)
        eval_acc = 0
        for data in eval_dataloader:
            img = data[0].to(device)
            out, _ = model(img)
            acc = 0
            words = 0
            for ot, tg in zip(out.cpu().transpose(0, 1), data[1]):
                tg = model_text(tg, Dataset.letters)
                ot = model_text(ot.squeeze(), Dataset.letters,
                                max_needed=True)
                rot = model_out_to_text(ot)
                for i, w in enumerate(tg):
                    words += 1
                    acc += int(w == rot[i] if len(rot) > i else False)
                print(tg, ot, rot)
            eval_acc = acc/words
            break
        print('accuracy:', eval_acc)

        wandb.log({"eval_acc": eval_acc, "batch_loss": batch_loss})

        scheduler.step()
        print('epoch_loss', losses[-1])
    wandb.finish()
    torch.save(model, config['model'] + '.pt')


if __name__ == '__main__':
    log = '--log' in sys.argv

    config = {
        'learning_rate': 0.0001,
        'epochs': 50,
        'train_size': 100*(10**4),
        'lr_step_size': 20,
        'lr_gamma': 0.8,
        'batch_size': 32,
        'model': 'v1',
        'dataset': 'mnt'
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train(config, log, device)
