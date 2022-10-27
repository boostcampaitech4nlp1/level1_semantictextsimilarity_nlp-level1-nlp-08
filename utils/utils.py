from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

def early_stop(monitor, patience, mode):
    early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
    return early_stop_callback

def best_save(save_path, top_k, monitor):
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=top_k, monitor=monitor, mode=mode)
    return checkpoint_callback

def get_checkpoint_callback(criterion, save_frequency, prefix="checkpoint", use_modelcheckpoint_filename=False):
    
    checkpoint_callback = None
    if criterion == 'step':
        checkpoint_callback = CheckpointEveryNSteps(save_frequency, prefix, use_modelcheckpoint_filename)
    elif criterion == 'epoch':
        checkpoint_callback = CheckpointEveryNEpochs(save_frequency, prefix, use_modelcheckpoint_filename)
    
    return checkpoint_callback

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """
    def __init__(
        self,
        save_step_frequency,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_epoch={epoch}_global_step={global_step}.ckpt"
            ckpt_path = os.path.join('model_save/', filename)
            trainer.save_checkpoint(ckpt_path)

class CheckpointEveryNEpochs(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """
    def __init__(
        self,
        save_epoch_frequency,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_epoch_frequency: how often to save in epochs
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_epoch_frequency = save_epoch_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if epoch % self.save_epoch_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_epoch={epoch}_global_step={global_step}.ckpt"
            ckpt_path = os.path.join('model_save/', filename)
            trainer.save_checkpoint(ckpt_path)
            
# 모니터링 할 쌍들
monitor_config = {
    "val_loss":{"monitor":"val_loss","mode":"min"},
    "val_pearson":{"monitor":"val_pearson","mode":"max"}
}
