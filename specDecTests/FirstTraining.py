from datasets import load_dataset;
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling;
import torch;

dataset_raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

#Tokenize function  is currently adjusted to this dataset
def tokenize(example):
    text = example["sentence"] + " => " + str(example["label"])

    enc = tokenizer(text, truncation=True)

    return enc

dataset_mapped = dataset_raw.map(tokenize, batched=False)

args = TrainingArguments(
    output_dir="./out",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_mapped["train"],
    eval_dataset=dataset_mapped["validation"],
    data_collator=data_collator
)

def eval_acc(model, tokenizer, dataset, n=200):
    model.eval()
    device = model.device
    correct = 0

    for i in range(n):
        ex = dataset[i]

        prompt = ex["sentence"] + " =>"
        inputs = tokenizer(prompt, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1)

        text = tokenizer.decode(out[0])

        pred = "1" if "1" in text[-3:] else "0"

        if str(ex["label"]) == pred:
            correct += 1

    return correct / n

# print("Before:", eval_acc(model, tokenizer, dataset_raw["validation"]))
trainer.train()
# print("After:", eval_acc(model, tokenizer, dataset_raw["validation"]))