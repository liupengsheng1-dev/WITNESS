import os
import re
import glob
import json
import hashlib
from io import BytesIO
from collections import OrderedDict

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

TASKS = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]

def clean_field(value):
    value = str(value).strip()
    value = value.replace("\t", " ")
    value = value.replace("\r", " ")
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def get_field(sample, candidate_names, default=None):
    for name in candidate_names:
        if name in sample and sample[name] is not None:
            return sample[name]
    return default

def normalize_category(category):
    category = clean_field(category).lower()
    category = category.replace("-", "_")
    category = category.replace(" ", "_")
    mapping = {
        "ocr": "OCR",
        "poster": "posters",
        "posters": "posters",
        "common_sense_reasoning": "commonsense_reasoning",
        "commonsense": "commonsense_reasoning",
        "commonsense_reasoning": "commonsense_reasoning",
        "numerical": "numerical_calculation",
        "numerical_calculation": "numerical_calculation",
        "translation": "text_translation",
        "text_translation": "text_translation",
        "code": "code_reasoning",
        "code_reasoning": "code_reasoning",
    }
    return mapping.get(category, category)

def decode_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
    if isinstance(image_value, str):
        return Image.open(image_value).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_value)}")

def get_image_id(sample, image, index):
    image_id = get_field(
        sample,
        ["image_id", "image_name", "file_name", "filename", "img_id", "image_path"],
    )
    if image_id is not None:
        return clean_field(image_id)

    image_hash = hashlib.sha1()
    image_hash.update(str(image.size).encode("utf-8"))
    image_hash.update(image.mode.encode("utf-8"))
    image_hash.update(image.tobytes())
    digest = image_hash.hexdigest()

    if digest:
        return digest
    return str(index)

def parse_yes_no(text):
    text = clean_field(text).lower()
    match = re.match(r"^\s*(yes|no)\b", text)
    if match is None:
        match = re.search(r"\b(yes|no)\b", text)
    if match is None:
        return "Unknown"
    return match.group(1).capitalize()

def validate_records(records):
    missing_tasks = [task for task in TASKS if task not in records or len(records[task]) == 0]
    if missing_tasks:
        raise RuntimeError("Missing MME categories: " + ", ".join(missing_tasks))

    for task in TASKS:
        for image_id, items in records[task].items():
            if len(items) != 2:
                raise RuntimeError(
                    f"{task}: image {image_id} has {len(items)} questions, expected 2."
                )

def write_mme_results(records, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    for task in TASKS:
        output_path = os.path.join(results_dir, f"{task}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for image_id, items in records[task].items():
                for item in items:
                    line = "\t".join(
                        [
                            clean_field(image_id),
                            clean_field(item["question"]),
                            clean_field(item["answer"]),
                            clean_field(item["prediction"]),
                        ]
                    )
                    f.write(line + "\n")

data_dir = "./data/MME"
parquet_files = sorted(glob.glob(os.path.join(data_dir, "test-*-of-00002.parquet")))
dataset = load_dataset(
    "parquet",
    data_files={"test": parquet_files},
)["test"]

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

save_path = "./result/janus_pro_mme/witness_alpha0.4_threshold0.8_scale1.jsonl"
results_dir = "./result/janus_pro_mme/witness_alpha0.4_threshold0.8_scale1"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

records = {task: OrderedDict() for task in TASKS}
correct_count = 0
total_count = 0

with open(save_path, "w", encoding="utf-8") as detail_file:
    for index in tqdm(range(len(dataset))):
        sample = dataset[index]

        question = get_field(sample, ["question", "text", "prompt"])
        answer = get_field(sample, ["answer", "label"])
        category = get_field(sample, ["category", "task", "type"])
        image_value = get_field(sample, ["image"])
        question_id = get_field(sample, ["question_id", "id"], default=index)

        if question is None:
            raise KeyError(f"Sample {index} has no question field.")
        if answer is None:
            raise KeyError(f"Sample {index} has no answer field.")
        if category is None:
            raise KeyError(f"Sample {index} has no category field.")
        if image_value is None:
            raise KeyError(f"Sample {index} has no image field.")

        question = clean_field(question)
        answer = parse_yes_no(answer)
        category = normalize_category(category)
        image = decode_image(image_value)
        image_id = get_image_id(sample, image, index)

        if category not in TASKS:
            raise ValueError(f"Unknown category: {category}, index={index}")

        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>\n" + question,
                "images": [image],
            },
            {
                "role": "<|Assistant|>",
                "content": "",
            },
        ]

        conversation_student = [
            {
                "role": "<|User|>",
                "content": question,
                "images": [],
            },
            {
                "role": "<|Assistant|>",
                "content": "",
            },
        ]

        inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        ).to("cuda")

        inputs_student = processor(
            conversations=conversation_student,
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

        full_prediction = tokenizer.batch_decode(
            generated_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
        )[0].strip()

        prediction = parse_yes_no(full_prediction)
        is_correct = prediction.lower() == answer.lower()

        correct_count += int(is_correct)
        total_count += 1
        acc = correct_count / total_count if total_count > 0 else 0

        if image_id not in records[category]:
            records[category][image_id] = []

        records[category][image_id].append(
            {
                "question_id": clean_field(question_id),
                "question": question,
                "answer": answer,
                "prediction": prediction,
            }
        )

        result = {
            "index": index,
            "image_id": image_id,
            "question_id": clean_field(question_id),
            "category": category,
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "full_prediction": full_prediction,
            "is_correct": is_correct,
        }

        detail_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        detail_file.flush()

        if not is_correct:
            print(
                f"[False] index={index}, "
                f"category={category}, "
                f"prediction={prediction}, "
                f"answer={answer}, "
                f"raw={full_prediction}"
            )

        print(f"Running accuracy: {correct_count}/{total_count} = {acc:.4f}")

validate_records(records)
write_mme_results(records, results_dir)

with open(save_path, "a", encoding="utf-8") as f:
    result = {"acc": f"{acc:.4f}"}
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
