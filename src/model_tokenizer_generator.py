from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch


def getT5Model_with_device(model_json_path: str, device: torch.device) -> T5ForConditionalGeneration:
    model_config = AutoConfig.from_pretrained(model_json_path)
    model = T5ForConditionalGeneration(config=model_config)
    model = model.to(device)
    return model


def getT5Tokenizer(token_path: str) -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(token_path)


def generate_result_by_device(model: T5ForConditionalGeneration,
                              tokenizer: T5Tokenizer,
                              input_text: str,
                              device: torch.device
                              ):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    return outputs


def get_and_save_model(model: str, save_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(str)
    model = AutoModelForCausalLM.from_pretrained(str)
    tokenizer.save_pretrained(str)
    model.save_pretrained(str)
