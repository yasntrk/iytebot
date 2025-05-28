from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import torch
import os


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def prepare_data(infile, outfile):
    with open(infile, 'r', encoding='utf-8') as fin, open(outfile, 'w', encoding='utf-8') as fout:
        for line in fin:
            pair = eval(line)  # json.loads satırına göre değiştirilebilir
            fout.write(f"<s> {pair['input']} {pair['output']} </s>\n")

prepare_data("chat_data.jsonl", "chat_dataset.txt")

# 3. TextDataset ile veriyi yükle
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )

train_dataset = load_dataset("chat_dataset.txt", tokenizer)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)


training_args = TrainingArguments(
    output_dir="./chatbot_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model("./chatbot_gpt2")
tokenizer.save_pretrained("./chatbot_gpt2")
