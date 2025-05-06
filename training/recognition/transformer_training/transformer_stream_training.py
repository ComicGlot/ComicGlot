import os
import pandas as pd
import torch
from evaluate import load
from datasets import Dataset, DatasetDict, load_dataset, IterableDatasetDict, IterableDataset
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
import random

torch.cuda.empty_cache()
gc.collect()

load_dotenv()
cer_metric = load("cer")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["FLASH_ATTENTION"] = "1"

SYNTH_OCR_DATA_PATH = os.path.abspath(os.getenv("SYNTH_OCR_DATA_PATH", "../synthtiger_gen/results"))
GT_FILE = os.path.join(SYNTH_OCR_DATA_PATH, "gt_shuffled_2.txt") #"gt_1.txt"
#GT_TRAIN_FILE = os.path.join(SYNTH_OCR_DATA_PATH, "gt_train.txt")
#GT_TEST_FILE = os.path.join(SYNTH_OCR_DATA_PATH, "gt_test.txt")

if not os.path.exists(GT_FILE):
    print(f"❌ ERROR: GT_FILE not found at {GT_FILE}")

try:
    gt_df = pd.read_csv(GT_FILE, delimiter="\t", header=None)
except Exception as e:
    print("Error reading CSV:", e)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GpuMemoryCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()

class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"].clone().detach() for item in batch])

        labels = torch.stack([item["labels"].clone().detach() for item in batch])
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
    
class Seq2SeqTrainerEvalSampling(Seq2SeqTrainer):
    def __init__(self, *args, eval_sample_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size = eval_sample_size
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Samples the evaluation dataset by shuffling and taking the first `eval_sample_size` elements.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if isinstance(eval_dataset, IterableDataset):
            eval_dataset = eval_dataset.shuffle(seed=random.randint(0, 9999)).take(self.eval_sample_size)

        return super().get_eval_dataloader(eval_dataset)
    
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

# image_transform = Compose([
#     Resize((384, 384)),
#     ToTensor(),
#     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

def preprocess_streamed_data(example):
    """
    Efficiently preprocesses images and text while keeping memory usage low.
    """
    image_path = example["file_path"]
    
    with Image.open(image_path).convert("RGB") as img:
        pixel_values = processor(img, return_tensors="pt")["pixel_values"][0].half()  # Convert to float16

    labels = processor.tokenizer(
        example["img_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )["input_ids"][0]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

MODEL_NAME = "team-lucid/trocr-small-korean"
#MODEL_NAME_2 = "./OCR/OCR_training/transformer_training/ko-trocr-cc100/2m_mecab_kotrocr_tuned/checkpoint-12000-v1"
#MODEL_NAME = "./OCR/OCR_training/transformer_training/ko-trocr-cc100/112k_split_words_1/checkpoint-7000"
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

# dataset = load_dataset(
#     "csv",
#     data_files={"train": GT_FILE},
#     split="train",
#     streaming=True,
#     delimiter="\t",
#     column_names=["file_path", "text"]
# )

dataset = load_dataset(
    "text",  # Load as raw text
    data_files={"train": GT_FILE},
    split="train",
    streaming=True
)

def process_line(example):
    parts = example["text"].split("\t", 1)  # Split into 2 parts
    return {"file_path": parts[0], "img_text": parts[1] if len(parts) > 1 else ""}

dataset = dataset.map(process_line, remove_columns=["text"])

# train_dataset = load_dataset(
#     "csv",
#     data_files={"train": GT_TRAIN_FILE},
#     split="train",
#     streaming=True,
#     delimiter="\t",
#     column_names=["file_path", "text"]
# )

# test_dataset = load_dataset(
#     "csv",
#     data_files={"test": GT_TEST_FILE},
#     split="test",
#     streaming=True,
#     delimiter="\t",
#     column_names=["file_path", "text"]
# )

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

#model.config.eos_token_id = processor.tokenizer.sep_token_id
# model.config.early_stopping = True
# model.config.no_repeat_ngram_size = 3
# #model.config.length_penalty = 2.0
# model.config.num_beams = 4
# model.config.max_length = 128

generation_config = GenerationConfig(
    max_new_tokens=128,
    no_repeat_ngram_size=3,
    early_stopping=True,
    num_beams=4
)

# model.generation_config = generation_config


def fix_image_path(example):
    example["file_path"] = os.path.join(DATA_PATH, example["file_path"])
    return example

dataset = dataset.map(fix_image_path, batched=False)
#test_dataset = test_dataset.map(fix_image_path, batched=False)
for i, sample in enumerate(dataset.take(1)):
    print(f"Train Sample {i+1}: {sample}")

dataset = dataset.map(preprocess_streamed_data, batched=False, remove_columns=["file_path", "img_text"])
#test_dataset = test_dataset.map(preprocess_streamed_data, batched=False, remove_columns=["file_path", "text"])
for i, sample in enumerate(dataset.take(1)):
    print(f"Train Sample {i+1}: {sample}")

#dataset = IterableDatasetDict({"train": train_dataset, "eval": test_dataset})

collator = CustomCollator(tokenizer)

training_args = Seq2SeqTrainingArguments(
    #remove_unused_columns=False,
    #eval_steps=200,
    #eval_strategy="steps",
    eval_strategy="no", #!!!!!!!!!!!!!!!!

    output_dir="./ko-trocr-cc100",
    per_device_train_batch_size=16, # 32
    per_device_eval_batch_size=16, # 32
    gradient_accumulation_steps=16, # 8

    logging_steps=100,
    save_steps=200,
    save_total_limit=16,  # Retain only the N most recent
    warmup_steps=6000,

    # num_train_epochs=3,
    max_steps=15_000, # If using `num_train_epochs`- not needed. 
    # max_steps * batch_size = processed imgs. (7000 * 16) + (24000 * 16) = 112 000 + 384 000 = 496 000

    lr_scheduler_type="cosine",  # Cosine decay (0.1)
    #lr_scheduler_type="linear",  # Simpler LR decay
    adam_beta1=0.9,
    adam_beta2=0.98,
    #learning_rate=5e-4,
    learning_rate=1e-4,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,

    #resume_from_checkpoint=MODEL_NAME_2 # !!!!!!!!!!! Restores optimizer & scheduler states
)

#trainer = Seq2SeqTrainerEvalSampling(
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    #eval_dataset=dataset["eval"],
    #eval_sample_size=512,
    #compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=collator,
    callbacks=[GpuMemoryCallback()],
)

trainer.train() # !!!!!!!!!!!!!!!!!!!!!!!!!! #resume_from_checkpoint=MODEL_NAME_2

model.save_pretrained("./trocr-finetuned")
tokenizer.save_pretrained("./trocr-finetuned")