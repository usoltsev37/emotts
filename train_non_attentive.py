import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.constants import CHECKPOINT_DIR, LOG_DIR
from src.data_process import VCTKBatch, VctkCollate, VCTKFactory
from src.models import NonAttentiveTacotron, NonAttentiveTacotronLoss
from src.train_config import TrainParams, load_config

MODEL_NAME = "model.pth"


def prepare_dataloaders(
    checkpoint: Path, config: TrainParams
) -> Tuple[DataLoader, DataLoader, int, int]:
    # Get data, data loaders and collate function ready
    phonemes_file = checkpoint / VCTKFactory.PHONEMES_JSON_NAME
    speakers_file = checkpoint / VCTKFactory.SPEAKER_JSON_NAME
    if os.path.isfile(phonemes_file):
        with open(phonemes_file, "r") as f:
            phonemes_to_id = json.load(f)
    else:
        phonemes_to_id = None
    if os.path.isfile(speakers_file):
        with open(speakers_file, "r") as f:
            speakers_to_id = json.load(f)
    else:
        speakers_to_id = None
    factory = VCTKFactory(
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        config=config.data,
        phonemes_to_id=phonemes_to_id,
        speakers_to_id=speakers_to_id,
    )
    factory.save_mapping(checkpoint)
    trainset, valset = factory.split_train_valid(config.test_size)
    collate_fn = VctkCollate()

    train_loader = DataLoader(
        trainset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        valset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
    )

    return (
        train_loader,
        val_loader,
        len(factory.phoneme_to_id),
        len(factory.speaker_to_id),
    )


def load_model(
    config: TrainParams, n_phonemes: int, n_speakers: int
) -> NonAttentiveTacotron:
    model = NonAttentiveTacotron(
        n_mel_channels=config.n_mels,
        n_phonems=n_phonemes,
        n_speakers=n_speakers,
        device=torch.device(config.device),
        config=config.model,
    )
    return model


def load_checkpoint(
    checkpoint_path: Path,
    model: NonAttentiveTacotron,
    optimizer: Adam,
    scheduler: StepLR,
) -> [NonAttentiveTacotron, Adam, StepLR]:
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
    return model, optimizer, scheduler


def save_checkpoint(
    filepath: Path, model: NonAttentiveTacotron, optimizer: Adam, scheduler: StepLR
):
    torch.save(
        {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        },
        filepath,
    )


def batch_to_device(batch: VCTKBatch, device: torch.device) -> VCTKBatch:
    batch_on_device = VCTKBatch(
        phonemes=batch.phonemes.to(device),
        num_phonemes=batch.num_phonemes,
        speaker_ids=batch.speaker_ids.to(device),
        durations=batch.durations.to(device),
        mels=batch.mels.to(device),
    )
    return batch_on_device


def validate(
    model: NonAttentiveTacotron,
    criterion: NonAttentiveTacotronLoss,
    val_loader: DataLoader,
    global_step: int,
    writer: SummaryWriter,
) -> None:

    model.eval()
    with torch.no_grad():

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            batch = batch_to_device(batch, model.device)
            durations, mel_outputs_postnet, mel_outputs = model(batch)
            loss = criterion(
                mel_outputs, mel_outputs_postnet, durations, batch.durations, batch.mels
            )
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss

        val_loss = val_loss / (i + 1)
        writer.add_scalar("Loss/valid", scalar_value=val_loss, global_step=global_step)

    model.train()


def train(config: TrainParams):

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    checkpoint_path = CHECKPOINT_DIR / config.checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / config.checkpoint_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, phonemes_count, speaker_count = prepare_dataloaders(
        checkpoint_path, config
    )
    model = load_model(config, phonemes_count, speaker_count)

    optimizer_config = config.optimizer
    optimizer = Adam(
        model.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.reg_weight,
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        eps=optimizer_config.adam_epsilon,
    )

    scheduler = StepLR(
        optimizer=optimizer,
        step_size=config.scheduler.decay_steps,
        gamma=config.scheduler.decay_rate,
    )

    criterion = NonAttentiveTacotronLoss(
        sample_rate=config.sample_rate,
        hop_size=config.hop_size,
        mels_weight=config.loss.mels_weight,
        duration_weight=config.loss.duration_weight,
    )

    iteration = 0
    epoch_offset = 0
    if os.path.isfile(checkpoint_path / MODEL_NAME):
        model, optimizer, scheduler = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    model.train()
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(config.device)

    for epoch in range(epoch_offset, config.epochs):
        for i, batch in enumerate(train_loader):
            global_step = epoch * len(train_loader) + iteration
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()
            durations, mel_outputs_postnet, mel_outputs = model(batch)

            loss = criterion(
                mel_outputs,
                mel_outputs_postnet,
                durations,
                batch.durations,
                batch.mels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_thresh)

            optimizer.step()
            if config.scheduler.start_decay <= i + 1 <= config.scheduler.last_epoch:
                scheduler.step()

            if global_step % config.log_steps == 0:
                writer.add_scalar("Loss/train", loss, global_step=global_step)

            if global_step % config.iters_per_checkpoint == 0:
                validate(
                    model=model,
                    criterion=criterion,
                    val_loader=val_loader,
                    global_step=global_step,
                    writer=writer,
                )
                save_checkpoint(
                    filepath=checkpoint_path / MODEL_NAME,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='configuration file path'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == '__main__':
    main()
