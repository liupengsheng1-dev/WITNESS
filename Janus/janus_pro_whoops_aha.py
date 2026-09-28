import os
import re
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

def normalize_answer(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

data_dir = "./data/WHOOPS-AHA"
dataset = load_dataset(
    "parquet",
    data_files={
        "train": [
            f"{data_dir}/train-00000-of-00002.parquet",
            f"{data_dir}/train-00001-of-00002.parquet",
        ]
    }
)
train_data = dataset["train"]

model = AutoModelForCausalLM.from_pretrained(
    "./models/Janus-Pro-7B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=False,
)

student_model = AutoModelForCausalLM.from_pretrained(
    "./models/Janus-Pro-7B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_safetensors=False,
)

processor = VLChatProcessor.from_pretrained("./models/Janus-Pro-7B")
tokenizer = processor.tokenizer

save_path = "./result/janus_pro_whoops_aha/witness_alpha0.4_adaptive0.8_1.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

batch_size = 1
correct_count = 0
total_count = 0

for start in tqdm(range(0, len(train_data), batch_size)):
    end = min(start + batch_size, len(train_data))
    batch_indices = []
    batch_samples = []

    for idx in range(start, end):
        batch_indices.append(idx)
        batch_samples.append(train_data[idx])

    batch_messages = []
    batch_messages_student = []
    batch_prompts = []

    for sample in batch_samples:
        prompt = "Output only the next word: " + sample["text"]
        batch_prompts.append(prompt)

        messages = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>\n" + prompt,
                "images": [sample["image"]],
            },
            {
                "role": "<|Assistant|>",
                "content": "",
            },
        ]

        messages_student = [
            {
                "role": "<|User|>",
                "content": prompt,
                "images": [],
            },
            {
                "role": "<|Assistant|>",
                "content": "",
            },
        ]

        batch_messages.append(messages)
        batch_messages_student.append(messages_student)

        batch_images = [sample["image"].convert("RGB") for sample in batch_samples]

        inputs = processor(
            conversations=batch_messages[0],
            images=batch_images,
            force_batchify=True,
        ).to("cuda")

        inputs_student = processor(
            conversations=batch_messages_student[0],
            images=[],
            force_batchify=True,
        ).to("cuda")

        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        inputs_embeds_student = student_model.prepare_inputs_embeds(**inputs_student)

    with torch.inference_mode():
        generated_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            inputs_embeds_student=inputs_embeds_student,
            enhance_decoding=True,
            student_model=student_model.language_model,
            tokenizer=tokenizer,
            max_new_tokens=8,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            do_sample=True,
            alpha=0.4,
            head_threshold=0.8,
            score_scale=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_texts = tokenizer.batch_decode(
        generated_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
    )
    output_texts = [x.strip() for x in output_texts]

    with open(save_path, "a", encoding="utf-8") as f:
        for idx, sample, prompt, output_text in zip(
            batch_indices,
            batch_samples,
            batch_prompts,
            output_texts,
        ):
            full_prediction = output_text.strip()
            counterfactual_tokens = sample.get("counterfactual_tokens", [])
            pred_norm = normalize_answer(full_prediction)
            counterfactual_tokens_norm = [
                normalize_answer(x) for x in counterfactual_tokens
            ]

            is_correct = any(
                pred_norm == ans or ans in pred_norm
                for ans in counterfactual_tokens_norm
            )

            correct_count += int(is_correct)
            total_count += 1
            acc = correct_count / total_count if total_count > 0 else 0

            if not is_correct:
                print(
                    f"[False] index={idx}, "
                    f"prediction={pred_norm}, "
                    f"counterfactual_tokens={counterfactual_tokens}, "
                    f"text={prompt}"
                )

                result = {
                    "index": idx,
                    "text": prompt,
                    "prediction": pred_norm,
                    "full_prediction": full_prediction,
                    "counterfactual_tokens": counterfactual_tokens,
                    "factual_tokens": sample.get("factual_tokens", None),
                    "is_correct": is_correct,
                }

                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Running accuracy: {correct_count}/{total_count} = {acc:.4f}")

with open(save_path, "a", encoding="utf-8") as f:
    result = {"acc": f"{acc:.4f}"}
    f.write(json.dumps(result, ensure_ascii=False) + "\n")