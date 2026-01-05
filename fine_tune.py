"""JAX/Flax fine-tuning CLI for Hugging Face models.

This script implements a modular JAX-only training loop for common text tasks
using Hugging Face Transformers and Optax. It supports configuration through
command-line flags or a YAML/JSON config file, handles dataset loading with the
`datasets` library, and provides simple checkpointing and evaluation hooks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import datasets as hf_datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from flax import linen as nn
from flax import struct
from flax.training import checkpoints, train_state
from jax import random as jax_random
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    FlaxAutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Argument parsing and config
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HF models with JAX/Flax")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None, help="Optional YAML/JSON config")
    parser.add_argument("--task_type", type=str, choices=["causal_lm", "sequence_classification"], default="causal_lm")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16")

    if args.config_file:
        args = merge_with_config(args, args.config_file)

    return args


def merge_with_config(args: argparse.Namespace, config_file: str) -> argparse.Namespace:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as err:  # pragma: no cover - defensive
            raise ImportError("Install pyyaml to load YAML configs") from err
        with path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    else:
        with path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)

    for key, value in config_data.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )


# ----------------------
# Reproducibility helpers
# ----------------------


def set_seed(seed: int) -> jax_random.KeyArray:
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    key = jax_random.PRNGKey(seed)
    logger.info("Seeds set to %s", seed)
    return key


def get_dtype(args: argparse.Namespace) -> jnp.dtype:
    if args.bf16:
        return jnp.bfloat16
    if args.fp16:
        return jnp.float16
    return jnp.float32


# -------------
# Data pipeline
# -------------


def load_datasets(args: argparse.Namespace) -> Dict[str, hf_datasets.Dataset]:
    if args.dataset_name:
        raw_datasets = hf_datasets.load_dataset(args.dataset_name)
    else:
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file
        if not data_files:
            raise ValueError("Provide a dataset name or train/validation files")
        extension = args.train_file.split(".")[-1]
        raw_datasets = hf_datasets.load_dataset(extension, data_files=data_files)

    if "validation" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(test_size=0.1, seed=args.seed)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]
    return raw_datasets


def tokenize_function(tokenizer: transformers.PreTrainedTokenizerBase, max_length: int, pad_to_max_length: bool, task_type: str):
    padding = "max_length" if pad_to_max_length else False

    def _tokenize(examples: Dict[str, Any]) -> Dict[str, Any]:
        text_column = examples.get("text") or examples.get("sentence")
        if text_column is None:
            raise KeyError("Dataset must contain a 'text' or 'sentence' column for tokenization")
        tokenized = tokenizer(
            text_column,
            padding=padding,
            truncation=True,
            max_length=max_length,
        )
        if task_type == "causal_lm":
            tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return _tokenize


def create_dataloader(dataset: hf_datasets.Dataset, batch_size: int, shuffle: bool = False) -> Iterable[Dict[str, np.ndarray]]:
    def _generator():
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch = dataset.select(batch_indices)
            yield {k: np.array(v) for k, v in batch.with_format("numpy")[:].items()}

    return _generator()


# -----------------------
# Model and train state
# -----------------------


def prepare_model_and_tokenizer(args: argparse.Namespace, raw_datasets: Dict[str, hf_datasets.Dataset]):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    dtype = get_dtype(args)

    if args.task_type == "sequence_classification":
        num_labels = getattr(raw_datasets["train"].features.get("label"), "num_classes", None) or config.num_labels
        config.num_labels = num_labels
        model = FlaxAutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            dtype=dtype,
        )
    else:
        model = FlaxAutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            dtype=dtype,
        )

    if args.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    return model, tokenizer


class TrainState(train_state.TrainState):
    dropout_rng: jax_random.KeyArray = struct.field(pytree_node=True)


def create_learning_rate_fn(num_train_steps: int, warmup_steps: int, learning_rate: float) -> Callable[[int], jnp.ndarray]:
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(num_train_steps - warmup_steps, 1),
        end_value=0.0,
    )
    return schedule_fn


def create_train_state(model: transformers.FlaxPreTrainedModel, learning_rate_fn: Callable[[int], jnp.ndarray], weight_decay: float, seed: int) -> TrainState:
    tx = optax.adamw(learning_rate=learning_rate_fn, weight_decay=weight_decay)
    rng = jax_random.PRNGKey(seed)
    return TrainState.create(apply_fn=model.__call__, params=model.params, tx=tx, dropout_rng=rng)


# -----------------------
# Training and evaluation
# -----------------------


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray, task_type: str) -> Dict[str, jnp.ndarray]:
    if task_type == "sequence_classification":
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)
        return {"accuracy": accuracy}
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits[..., :-1, :], labels[..., 1:]).mean()
        perplexity = jnp.exp(loss)
        return {"loss": loss, "perplexity": perplexity}


def train_step(model, state: TrainState, batch: Dict[str, np.ndarray], task_type: str):
    dropout_rng, new_dropout_rng = jax_random.split(state.dropout_rng)

    def loss_fn(params):
        outputs = model(**batch, params=params, dropout_rng=dropout_rng, train=True)
        logits = outputs.logits
        if task_type == "sequence_classification":
            labels = batch["labels"]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        else:
            labels = batch["labels"]
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    metrics = compute_metrics(logits, batch["labels"], task_type)
    metrics["loss"] = loss
    return new_state, metrics


def eval_step(model, params, batch: Dict[str, np.ndarray], task_type: str):
    outputs = model(**batch, params=params, train=False)
    logits = outputs.logits
    metrics = compute_metrics(logits, batch["labels"], task_type)
    return metrics


def train(args: argparse.Namespace):
    setup_logging()
    rng = set_seed(args.seed)

    raw_datasets = load_datasets(args)
    model, tokenizer = prepare_model_and_tokenizer(args, raw_datasets)

    tokenized_datasets = raw_datasets.map(
        tokenize_function(tokenizer, args.max_seq_length, args.pad_to_max_length, args.task_type),
        batched=True,
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in {"label"}],
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    steps_per_epoch = math.ceil(len(train_dataset) / args.per_device_train_batch_size)
    num_train_steps = args.max_train_steps or steps_per_epoch * args.num_train_epochs

    learning_rate_fn = create_learning_rate_fn(num_train_steps, args.warmup_steps, args.learning_rate)
    state = create_train_state(model, learning_rate_fn, args.weight_decay, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(train_dataset, args.per_device_train_batch_size, shuffle=True)
    eval_loader = create_dataloader(eval_dataset, args.per_device_eval_batch_size, shuffle=False)

    p_train_step = jax.jit(lambda st, bt: train_step(model, st, bt, args.task_type))
    p_eval_step = jax.jit(lambda params, bt: eval_step(model, params, bt, args.task_type))

    global_step = 0
    for epoch in range(args.num_train_epochs):
        for batch in train_loader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            state, metrics = p_train_step(state, batch)
            global_step += 1

            if global_step % args.logging_steps == 0:
                logger.info(
                    "Epoch %s step %s - loss: %.4f", epoch + 1, global_step, metrics.get("loss", 0.0)
                )

            if global_step % args.eval_steps == 0:
                eval_metrics = run_evaluation(p_eval_step, state.params, eval_loader)
                logger.info("Eval at step %s: %s", global_step, {k: float(v) for k, v in eval_metrics.items()})

            if global_step % args.save_steps == 0:
                save_checkpoint(output_dir, state, global_step)

            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    save_checkpoint(output_dir, state, global_step)

    if args.push_to_hub:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    return state


def run_evaluation(eval_fn: Callable, params: Dict[str, Any], dataloader: Iterable[Dict[str, np.ndarray]]) -> Dict[str, float]:
    aggregated = {}
    count = 0
    for batch in dataloader:
        batch = {k: jnp.array(v) for k, v in batch.items()}
        metrics = eval_fn(params, batch)
        metrics = {k: float(v) for k, v in metrics.items()}
        for k, v in metrics.items():
            aggregated[k] = aggregated.get(k, 0.0) + v
        count += 1
    return {k: v / max(count, 1) for k, v in aggregated.items()}


def save_checkpoint(output_dir: Path, state: TrainState, step: int) -> None:
    checkpoints.save_checkpoint(
        ckpt_dir=output_dir,
        target={"params": state.params, "opt_state": state.opt_state},
        step=step,
        overwrite=True,
    )
    logger.info("Saved checkpoint at step %s to %s", step, output_dir)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
