import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from datasets import load_dataset


DEVICE = "cuda:4"
PCI_BUS_ID=4
CUDA_VISIBLE_DEVICES=4
USE_LORA = False
USE_QLORA = True
model_id = "HuggingFaceM4/Idefics3-8B-Llama3"

processor = AutoProcessor.from_pretrained(
    model_id
)


if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|connector).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,            
        )
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="flash_attention_2",
        #device_map="auto"
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to(DEVICE)

for param in model.model.vision_model.parameters():
    param.requires_grad = False

ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
split_ds = ds["validation"].train_test_split(test_size=0.8)
train_ds = split_ds["train"]

image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
  texts = []
  images = []
  for example in examples:
      image = example["image"]
      question = example["question"]
      answer = example["multiple_choice_answer"]
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "Answer briefly."},
                  {"type": "image"},
                  {"type": "text", "text": question}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": answer}
              ]
          }
      ]
      text = processor.apply_chat_template(messages, add_generation_prompt=False)
      texts.append(text.strip())
      images.append([image])

  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  labels = batch["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  labels[labels == image_token_id] = -100 
  batch["labels"] = labels

  return batch

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    #evaluation_strategy="epoch",
    bf16=True,
    output_dir="./idefics3-llama-vqav2",
    hub_model_id="idefics3-llama-vqav2",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)

trainer.train()
trainer.push_to_hub()