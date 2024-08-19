import pandas as pd
from datasets import Dataset
from transformers import CanineTokenizer, CanineForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import multiprocessing

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

set_seed(42) 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()
torch.cuda.memory.set_per_process_memory_fraction(0.9) 

project_dir = os.path.dirname(os.path.abspath(__file__))

data_source = os.path.join(project_dir, '..', 'data', 'saudi_privacy_policy')

train_df = pd.read_csv(os.path.join(data_source, "train.csv"), header=None, names=['label', 'text'])
test_df = pd.read_csv(os.path.join(data_source, "test.csv"), header=None, names=['label', 'text'])
train_df['label']=train_df['label']-1
test_df['label']=test_df['label']-1

val_df = pd.read_csv(os.path.join(data_source, "val.csv"), header=None, names=['label', 'text'])
val_df['label'] = val_df['label'] - 1

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)
val_ds = Dataset.from_pandas(val_df)

# 2. Define labels and mappings (assuming 10 labels, 1-10)
labels = [str(i) for i in range(1, 11)]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# 3. Tokenize (use the 'text' column from your dataset)
tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
train_ds = train_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True), batched=True)
train_ds.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
train_ds = train_ds.rename_column(original_column_name="label", new_column_name="labels")

test_ds = test_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True), batched=True)
test_ds.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
test_ds = test_ds.rename_column(original_column_name="label", new_column_name="labels")

val_ds = val_ds.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True), batched=True)
val_ds.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
val_ds = val_ds.rename_column(original_column_name="label", new_column_name="labels")


# reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# 5. Create DataLoaders
train_dataloader = DataLoader(train_ds, batch_size=16, num_workers=19, worker_init_fn=seed_worker)
test_dataloader = DataLoader(test_ds, batch_size=16, num_workers=19, worker_init_fn=seed_worker)
val_dataloader = DataLoader(val_ds, batch_size=16, num_workers=19, worker_init_fn=seed_worker)


import pytorch_lightning as pl
from transformers import CanineForSequenceClassification, AdamW
import torch.nn as nn

class CanineClassifier(pl.LightningModule):
    def __init__(self, num_labels=10, learning_rate=5e-5, weight_decay=0.01, id2label=None):
        super(CanineClassifier, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        if id2label is None:
            id2label = {i: str(i) for i in range(num_labels)}
        self.id2label = id2label
        self.model = CanineForSequenceClassification.from_pretrained('google/canine-s',
                                                                     num_labels=len(labels),
                                                                     id2label=id2label,
                                                                     label2id=label2id,
                                                                     hidden_dropout_prob=0.3,
                                                                     attention_probs_dropout_prob=0.3)
        self.test_step_outputs = []

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits

        predictions = logits.argmax(-1)
        correct = (predictions == batch['labels']).sum().item()
        accuracy = correct/batch['input_ids'].shape[0]

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_accuracy", accuracy, on_step=True, on_epoch=True)

        logits = self(batch['input_ids'], batch['attention_mask'], batch['token_type_ids']).logits
        preds = torch.argmax(logits, dim=1)
        labels = batch['labels']

        # Accumulate predictions and labels
        self.test_step_outputs.append({"preds": preds, "labels": labels})

        return {"test_loss": loss, "test_accuracy": accuracy}
    def on_test_epoch_start(self):
        self.test_step_outputs = []
    
    def on_test_epoch_end(self):
        preds = torch.cat([x["preds"] for x in self.test_step_outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.test_step_outputs]).cpu().numpy()

        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - CANINE')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix - CANINE.png')
        plt.close()

        # Print confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Compute classification report
        cr = classification_report(labels, preds, target_names=list(self.id2label.values()))
        print("Classification Report:")
        print(cr)

        # Clear the outputs list
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation_loss",
            },
        }

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
    def test_dataloader(self):
        return test_dataloader

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Early stopping callback
    from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
    early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='min'
    )


    # Early stopping callback
    from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_accuracy',
        dirpath='checkpoints',
        filename=f'canine-best-checkpoint' + '-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max'
    )

    import wandb

    wandb.login()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger


    model = CanineClassifier(num_labels=10, id2label=id2label)

    wandb_logger = WandbLogger(name='canine-s', project='CANINE')
    trainer = Trainer( logger=wandb_logger, max_epochs=50, callbacks=[checkpoint_callback], deterministic=True,)
    trainer.fit(model)


    # Test the model
    best_model = CanineClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    test_result = trainer.test(model)
    print(f"Test result: {test_result}")

    test_result = trainer.test(best_model)
    print(f"Best Model Test result: {test_result}")



    model.model.save_pretrained('canine')



