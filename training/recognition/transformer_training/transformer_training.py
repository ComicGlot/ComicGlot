import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForVision2Seq,
    TrOCRProcessor,
    TrainerCallback,
    GenerationConfig,
)
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from dotenv import load_dotenv
from torch.utils.data import default_collate
import gc

torch.cuda.empty_cache()
gc.collect()

load_dotenv()

DATA_PATH = os.path.abspath(os.getenv("DATA_PATH", "../synthtiger_gen/results"))
TOKENIZER_PATH = os.path.abspath(os.getenv("TOKENIZER_PATH", "../tokenizer_training/ko_trocr_cc100_tokenizer"))
#IMAGES_PATH = os.path.join(DATA_PATH, "images")
GT_FILE = os.path.join(DATA_PATH, "gt_1.txt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_custom_tokenizer = False

class GpuMemoryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pixel_values = torch.stack([torch.tensor(item["pixel_values"]) for item in batch])

        labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

def load_dataset(gt_file, images_path, sample_size=None):
    """
    Load and preprocess the dataset
    """

    df = pd.read_csv(gt_file, sep="\t", header=None, names=["file_path", "text"])
    df["file_path"] = df["file_path"].apply(lambda x: os.path.abspath(os.path.join(images_path, x)))

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return DatasetDict({"train": train_dataset, "test": test_dataset})

image_transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_data(examples):
    """
    Preprocess images and tokenize text
    """
    image_path = examples["file_path"]
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image)

    labels = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )["input_ids"][0]

    return {
        "pixel_values": image,
        "labels": labels,
    }

def preprocess_proc_data(examples):
    """
    Batched preprocessing for images and text.
    """
    images = [Image.open(path).convert("RGB") for path in examples["file_path"]]
    pixel_values = processor(images, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )["input_ids"]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

def preprocess_batched_data(examples):
    """
    Preprocess batched images and tokenize text
    """
    image_paths = examples["file_path"]  # List of image paths
    images = [image_transform(Image.open(path).convert("RGB")).numpy() for path in image_paths]


    labels = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )["input_ids"][0]

    return {
        "pixel_values": images,
        "labels": labels,
    }

# MODEL_NAME = "microsoft/trocr-base-stage1"
#MODEL_NAME = "team-lucid/trocr-small-korean"
MODEL_NAME = "./OCR/OCR_training/transformer_training/ko-trocr-cc100/112k_split_words_1/checkpoint-7000"
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

dataset = load_dataset(GT_FILE, DATA_PATH, sample_size=30_000) # Set sample_size if not full dataset is used

if use_custom_tokenizer:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(TOKENIZER_PATH, "tokenizer.json"),
        #merges_file=os.path.join(TOKENIZER_PATH, "merges.txt"),
        #vocab_file=os.path.join(TOKENIZER_PATH, "vocab.json"),
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<s>",
        unk_token="<unk>"
    )

    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.decoder.resize_token_embeddings(len(tokenizer)) 

    dataset = dataset.map(preprocess_data, batched=False)
    # dataset = dataset.map(preprocess_batched_data, batched=True, batch_size=100)

else:
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    tokenizer = processor.tokenizer

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.max_length = 128
    dataset = dataset.map(preprocess_proc_data, batched=True)

print(dataset)
trainer_dataset = dataset.remove_columns(['file_path', 'text'])
print("Training samples:", len(trainer_dataset["train"]))
print("Validation samples:", len(trainer_dataset["test"]))

# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     padding=True
# )   
 
collator = CustomCollator(tokenizer)

training_args = Seq2SeqTrainingArguments(
    #remove_unused_columns=False,

    eval_strategy="steps",
    output_dir="./ko-trocr-cc100",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,

    logging_steps=100,
    save_steps=200,
    save_total_limit=2,  # Retain only the N most recent
    eval_steps=200,
    warmup_steps=4000,

    num_train_epochs=3,
    # max_steps=100_000, # If using `max_steps`- not needed

    #lr_scheduler_type="cosine",  # Cosine decay (0.1)
    lr_scheduler_type="linear",  # Simpler LR decay
    adam_beta1=0.9,
    adam_beta2=0.98,
    learning_rate=2e-4,
    #learning_rate=1e-4,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=trainer_dataset["train"],
    eval_dataset=trainer_dataset["test"],
    tokenizer=tokenizer,
    data_collator=collator,
    callbacks=[GpuMemoryCallback()],
)

trainer.train()

model.save_pretrained("./trocr-finetuned")
tokenizer.save_pretrained("./trocr-finetuned")
