from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def early_stop(monitor, patience, mode):
    early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
    return early_stop_callback

def best_save(save_path, top_k, monitor):
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, save_top_k=top_k, monitor=monitor)
    return checkpoint_callback