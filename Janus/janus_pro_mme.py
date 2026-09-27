import os
import re
import glob
import json
import hashlib
import argparse
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

    raise TypeError(
        f"Unsupported image type: {type(image_value)}"
    )


def get_image_id(sample, image, index):
    image_id = get_field(
        sample,
        [
            "image_id",
            "image_name",
            "file_name",
            "filename",
            "img_id",
            "image_path",
        ],
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
    missing_tasks = [
        task
        for task in TASKS
        if task not in records or len(records[task]) == 0
    ]

    if missing_tasks:
        raise RuntimeError(
            "以下 MME 类别没有生成结果："
            + ", ".join(missing_tasks)
        )

    for task in TASKS:
        for image_id, items in records[task].items():
            if len(items) != 2:
                raise RuntimeError(
                    f"{task} 中图像 {image_id} 对应 "
                    f"{len(items)} 个问题，正常应为 2 个。"
                )


def write_mme_results(records, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    for task in TASKS:
        output_path = os.path.join(
            results_dir,
            f"{task}.txt",
        )

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["baseline", "enhance"],
        default="enhance",
    )

    parser.add_argument(
        "--data_dir",
        default="/data/research_users/liupengsheng/asset/MME",
    )

    parser.add_argument(
        "--model_path",
        default=(
            "/data/research_users/liupengsheng/models/"
            "Janus-Pro-7B"
        ),
    )

    parser.add_argument(
        "--output_root",
        default=(
            "/data/research_users/liupengsheng/"
            "DeCK-image/result/janus_pro_mme"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--head_threshold",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--score_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    parquet_files = sorted(
        glob.glob(
            os.path.join(
                args.data_dir,
                "test-*-of-00002.parquet",
            )
        )
    )

    print("找到的 parquet 文件：")

    for path in parquet_files:
        print(path)

    if len(parquet_files) != 2:
        raise FileNotFoundError(
            "完整 MME 应包含两个 parquet 文件，"
            f"当前找到 {len(parquet_files)} 个。"
        )

    dataset = load_dataset(
        "parquet",
        data_files={
            "test": parquet_files,
        },
    )["test"]

    print(f"数据总量：{len(dataset)}")
    print(f"数据字段：{dataset.column_names}")

    first_sample_info = {
        key: dataset[0][key]
        for key in dataset.column_names
        if key != "image"
    }

    print("第一条样本：")
    print(first_sample_info)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=False,
    )
    model.eval()

    student_model = None

    if args.mode == "enhance":
        student_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_safetensors=False,
        )
        student_model.eval()

    processor = VLChatProcessor.from_pretrained(
        args.model_path
    )
    tokenizer = processor.tokenizer

    if args.mode == "baseline":
        experiment_name = "baseline"
    else:
        experiment_name = (
            f"enhance_alpha{args.alpha}_"
            f"threshold{args.head_threshold}_"
            f"scale{args.score_scale}"
        )

    results_dir = os.path.join(
        args.output_root,
        experiment_name,
    )

    detail_path = os.path.join(
        args.output_root,
        f"{experiment_name}.jsonl",
    )

    os.makedirs(args.output_root, exist_ok=True)

    records = {
        task: OrderedDict()
        for task in TASKS
    }

    correct_count = 0
    total_count = 0

    with open(
        detail_path,
        "w",
        encoding="utf-8",
    ) as detail_file:

        progress_bar = tqdm(
            range(len(dataset)),
            desc=f"Janus-Pro MME {args.mode}",
        )

        for index in progress_bar:
            sample = dataset[index]

            question = get_field(
                sample,
                ["question", "text", "prompt"],
            )

            answer = get_field(
                sample,
                ["answer", "label"],
            )

            category = get_field(
                sample,
                ["category", "task", "type"],
            )

            image_value = get_field(
                sample,
                ["image"],
            )

            question_id = get_field(
                sample,
                ["question_id", "id"],
                default=index,
            )

            if question is None:
                raise KeyError(
                    f"第 {index} 条数据没有问题字段。"
                )

            if answer is None:
                raise KeyError(
                    f"第 {index} 条数据没有答案字段。"
                )

            if category is None:
                raise KeyError(
                    f"第 {index} 条数据没有类别字段。"
                )

            if image_value is None:
                raise KeyError(
                    f"第 {index} 条数据没有图像字段。"
                )

            question = clean_field(question)
            answer = parse_yes_no(answer)
            category = normalize_category(category)
            image = decode_image(image_value)

            image_id = get_image_id(
                sample,
                image,
                index,
            )

            if category not in TASKS:
                raise ValueError(
                    f"发现未知类别：{category}，index={index}"
                )

            conversation = [
                {
                    "role": "<|User|>",
                    "content": (
                        "<image_placeholder>\n"
                        + question
                    ),
                    "images": [image],
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

            inputs_embeds = model.prepare_inputs_embeds(
                **inputs
            )

            if args.mode == "enhance":
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

                inputs_student = processor(
                    conversations=conversation_student,
                    images=[],
                    force_batchify=True,
                ).to("cuda")

                inputs_embeds_student = (
                    student_model.prepare_inputs_embeds(
                        **inputs_student
                    )
                )

                with torch.inference_mode():
                    generated_ids = (
                        model.language_model.generate(
                            inputs_embeds=inputs_embeds,
                            inputs_embeds_student=(
                                inputs_embeds_student
                            ),
                            enhance_decoding=True,
                            student_model=(
                                student_model.language_model
                            ),
                            tokenizer=tokenizer,
                            max_new_tokens=(
                                args.max_new_tokens
                            ),
                            top_p=0.001,
                            top_k=1,
                            temperature=0.01,
                            do_sample=True,
                            alpha=args.alpha,
                            head_threshold=(
                                args.head_threshold
                            ),
                            score_scale=args.score_scale,
                            pad_token_id=(
                                tokenizer.eos_token_id
                            ),
                            eos_token_id=(
                                tokenizer.eos_token_id
                            ),
                        )
                    )

            else:
                with torch.inference_mode():
                    generated_ids = (
                        model.language_model.generate(
                            inputs_embeds=inputs_embeds,
                            max_new_tokens=(
                                args.max_new_tokens
                            ),
                            do_sample=False,
                            pad_token_id=(
                                tokenizer.eos_token_id
                            ),
                            eos_token_id=(
                                tokenizer.eos_token_id
                            ),
                        )
                    )

            full_prediction = tokenizer.batch_decode(
                generated_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )[0].strip()

            prediction = parse_yes_no(
                full_prediction
            )

            is_correct = (
                prediction.lower()
                == answer.lower()
            )

            correct_count += int(is_correct)
            total_count += 1

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

            detail_result = {
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

            detail_file.write(
                json.dumps(
                    detail_result,
                    ensure_ascii=False,
                )
                + "\n"
            )
            detail_file.flush()

            running_acc = (
                correct_count / total_count
                if total_count > 0
                else 0
            )

            progress_bar.set_postfix(
                accuracy=f"{running_acc:.4f}"
            )

            if not is_correct:
                tqdm.write(
                    f"[False] index={index}, "
                    f"category={category}, "
                    f"prediction={prediction}, "
                    f"answer={answer}, "
                    f"raw={full_prediction}"
                )

    validate_records(records)

    write_mme_results(
        records,
        results_dir,
    )

    print("\n各类别问题数量：")

    for task in TASKS:
        question_num = sum(
            len(items)
            for items in records[task].values()
        )

        image_num = len(records[task])

        print(
            f"{task}: "
            f"{image_num} images, "
            f"{question_num} questions"
        )

    final_acc = (
        correct_count / total_count
        if total_count > 0
        else 0
    )

    print(
        f"\n普通逐问题准确率："
        f"{correct_count}/{total_count} "
        f"= {final_acc:.4f}"
    )

    print(f"官方评测结果目录：{results_dir}")
    print(f"详细预测文件：{detail_path}")


if __name__ == "__main__":
    main()