from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch


def getT5Model_with_device(
    model_json_path: str, device: torch.device
) -> T5ForConditionalGeneration:
    model_config = AutoConfig.from_pretrained(model_json_path)
    model = T5ForConditionalGeneration(config=model_config)
    model = model.to(device)
    return model


def get_other_model_with_device(
    model_json_path: str, device: torch.device
) -> T5ForConditionalGeneration:
    model_config = AutoConfig.from_pretrained(model_json_path)
    model = AutoModelForCausalLM.from_config(config=model_config)
    model = model.to(device)
    return model


def getT5Tokenizer(token_path: str) -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(token_path)


def get_other_tokenizer(token_path: str) -> T5Tokenizer:
    return AutoTokenizer.from_pretrained(token_path)


def generate_result_by_T5(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    input_text: str,
    device: torch.device,
):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    return outputs


def generate_result_by_other_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    device: torch.device,
):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    return outputs


def get_and_save_T5_model(model: str, save_dir: str):
    tokenizer = T5Tokenizer.from_pretrained(model)
    model = T5ForConditionalGeneration.from_pretrained(model)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)


def get_and_save_other_model(model: str, save_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    return tokenizer, model
