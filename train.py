import argparse
import re
import json
import os
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import numpy as np
from sklearn.model_selection import train_test_split

# Отключаем некоторые предупреждения
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")

# -------------------- Утилиты --------------------

def read_json_or_jsonl(path: str) -> List[Dict[str, str]]:

    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if not text:
                return []
            
            # Пробуем как JSON массив
            if text.startswith('['):
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    pass
            
            # Пробуем как JSONL
            items = []
            with open(path, 'r', encoding='utf-8') as fr:
                for i, line in enumerate(fr, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Ошибка в строке {i}: {e}")
                        continue
            
            return items
    except Exception as e:
        print(f"Ошибка чтения файла {path}: {e}")
        return []

def get_default_lora_targets(model_name: str) -> List[str]:
    """Получение целевых модулей для LoRA с поддержкой большего количества архитектур."""
    mn = model_name.lower()
    
    # Llama/Mistral архитектуры
    if 'llama' in mn or 'mistral' in mn or 'alpaca' in mn or 'vicuna' in mn:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # GPT архитектуры
    if 'gpt2' in mn or 'gpt-neo' in mn or 'gpt-j' in mn or 'gpt' in mn or 'dialo' in mn:
        return ["c_attn", "c_proj", "c_fc"]
    
    # Qwen архитектуры
    if 'qwen' in mn:
        return ["c_attn", "c_proj", "w1", "w2"]
    
    # Bloom архитектуры
    if 'bloom' in mn:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # MPT архитектуры
    if 'mpt' in mn:
        return ["Wqkv", "out_proj", "up_proj", "down_proj"]
    
    # OPT архитектуры
    if 'opt' in mn:
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    
    # Общие модули трансформеров
    return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# -------------------- Очистка текста --------------------

def clean_text(text):

    if not text or not isinstance(text, str):
        return ""
    
    # Удаляем URL
    text = re.sub(r'\b(?:https?://|www\.)\S+\b', '', text)
    
    # Нормализуем медицинские термины
    medical_terms = {
        r'\b(?:Chat ?Doctor|ChatDoctor|Virtual ?Doc)\b': 'Doctor',
        r'\bDr\.\b': 'Doctor',
        r'\bMD\b': 'Doctor',
        r'\bpt\.\b': 'patient',`
        r'\bpt\b': 'patient'
    }
    
    for pattern, replacement in medical_terms.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Удаляем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем лишние знаки препинания
    text = re.sub(r'[.,;:!?]{2,}', lambda m: m.group()[0], text)
    
    return text.strip()

# -------------------- Кастомные коллбэки --------------------

class GradientMonitorCallback(TrainerCallback):
    """Мониторинг градиентов для предотвращения исчезновения/взрыва градиентов."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Пытаемся получить градиенты, если они доступны
            try:
                model = kwargs.get('model')
                if model is not None and hasattr(model, 'parameters'):ёёёё
                    total_norm = 0.0
                    has_gradients = False
                    
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm += param_norm ** 2
                            has_gradients = True
                    
                    if has_gradients:
                        total_norm = total_norm ** 0.5
                        logs['grad_norm'] = total_norm
                        
                        # Логируем предупреждение о слишком больших градиентах
                        if total_norm > 10.0:
                            logs['grad_warning'] = f"Большой градиент: {total_norm:.2f}"
            except Exception as e:
                # Игнорируем ошибки мониторинга градиентов
                pass

class LearningRateMonitorCallback(TrainerCallback):
    """Мониторинг learning rate с безопасной обработкой."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            try:
                # Пробуем разные способы получить learning rate
                if hasattr(state, 'scheduler') and state.scheduler is not None:
                    lr = state.scheduler.get_last_lr()[0]
                elif hasattr(state, 'optimizer') and state.optimizer is not None:
                    # Берем LR из первого параметра оптимизатора
                    for param_group in state.optimizer.param_groups:
                        if 'lr' in param_group:
                            lr = param_group['lr']
                            break
                    else:
                        lr = args.learning_rate
                else:
                    lr = args.learning_rate
                
                logs['learning_rate'] = lr
            except (AttributeError, IndexError, KeyError) as e:
                # В случае ошибки используем значение по умолчанию
                logs['learning_rate'] = args.learning_rate
                logs['lr_warning'] = f"Не удалось получить LR: {str(e)}"

class TrainingProgressCallback(TrainerCallback):
    """Коллбэк для отслеживания прогресса обучения."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            print(f"Шаг {state.global_step}/{state.max_steps} "
                  f"({100.0 * state.global_step / state.max_steps:.1f}%) - "
                  f"Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nЭпоха {state.epoch:.1f} завершена. "
              f"Общее время: {state.global_step / state.num_train_samples * state.max_steps / 3600:.2f} ч")

# -------------------- Подготовка данных --------------------

def create_chat_prompt(messages, tokenizer):
    try:
        # Пробуем использовать встроенный чат-темплейт токенизатора
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            return text
    except Exception as e:
        print(f"Ошибка применения chat template: {e}")
    
    # Fallback: ручной формат
    formatted = []
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content']
        formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)

def prepare_dataset_for_chat(
    items: List[Dict[str, str]], 
    tokenizer: AutoTokenizer, 
    max_length: int = 1024,
    val_split: float = 0.1
):

    if not items:
        return None, None
    
    train_records = []
    val_records = []
    
    # Разделяем данные на train/val
    if len(items) > 1:
        train_items, val_items = train_test_split(
            items, 
            test_size=val_split, 
            random_state=42,
            shuffle=True
        )
    else:
        # Если только один элемент, используем его для тренировки
        train_items, val_items = items, []
    
    print(f"Train samples: {len(train_items)}, Val samples: {len(val_items)}")
    
    for split_name, split_items in [("train", train_items), ("val", val_items)]:
        records = []
        for it in split_items:
            instruction = clean_text(it.get('instruction', 
                'You are a helpful medical assistant. Provide accurate and professional responses to medical questions.'))
            user_input = clean_text(it.get('input', ''))
            output = clean_text(it.get('output', ''))
            
            # Пропускаем пустые примеры
            if not output and not user_input:
                continue
            
            # Создаем сообщения
            messages = []
            if instruction:
                messages.append({"role": "system", "content": instruction})
            if user_input:
                messages.append({"role": "user", "content": user_input})
            if output:
                messages.append({"role": "assistant", "content": output})
            
            # Создаем промпт
            text = create_chat_prompt(messages, tokenizer)
            text += tokenizer.eos_token
            
            # Токенизация с учетом максимальной длины
            tokenized = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length, 
                padding=False,
                return_tensors=None
            )
            
            if len(tokenized["input_ids"]) < 10:  # Пропускаем слишком короткие примеры
                continue
            
            input_ids = tokenized["input_ids"]
            

            decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
            

            assistant_pos = len(input_ids)
            if "assistant" in decoded.lower():
                # Находим позицию "assistant" в токенах
                assistant_tokens = tokenizer.encode("assistant", add_special_tokens=False)
                for i in range(len(input_ids) - len(assistant_tokens) + 1):
                    if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
                        # Ищем конец роли (обычно ": " или "\n")
                        for j in range(i + len(assistant_tokens), min(len(input_ids), i + len(assistant_tokens) + 5)):
                            if tokenizer.decode([input_ids[j]]) in [":", "\n", " "]:
                                assistant_pos = j + 1
                                break
                        break
            
            # Создаем labels (mask до ответа ассистента)
            labels = [-100] * assistant_pos + input_ids[assistant_pos:]
            
            # Проверяем, что есть что обучать
            if sum(1 for l in labels if l != -100) < 5:
                continue
            
            record = {
                "input_ids": input_ids,
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            }
            
            records.append(record)
        
        if split_name == "train":
            train_records = records
        else:
            val_records = records
    
    train_dataset = Dataset.from_list(train_records) if train_records else None
    val_dataset = Dataset.from_list(val_records) if val_records else None
    
    return train_dataset, val_dataset

def prepare_dataset_from_hf(
    dataset_name: str, 
    tokenizer: AutoTokenizer, 
    max_samples: Optional[int] = None, 
    max_length: int = 1024,
    val_split: float = 0.1
):
    """Загрузка датасета из HuggingFace с улучшенной обработкой."""
    print(f"Загрузка датасета: {dataset_name}")
    
    try:

        dataset = load_dataset(dataset_name, cache_dir="./cache")
        
        # Извлекаем train split
        if "train" in dataset:
            dataset = dataset["train"]
        elif isinstance(dataset, dict):
            # Берем первый доступный сплит
            first_key = list(dataset.keys())[0]
            dataset = dataset[first_key]
        
        # Ограничиваем количество сэмплов
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"Используем {max_samples} сэмплов из датасета")
        else:
            print(f"Используем все {len(dataset)} сэмплов из датасета")
        
        # Преобразуем в нужный формат
        items = []
        for item in dataset:
            # Поддерживаем различные форматы данных
            if 'messages' in item:  # Chat format
                messages = item['messages']
                instruction = ""
                user_input = ""
                output = ""
                
                for msg in messages:
                    if msg['role'] == 'system':
                        instruction = msg['content']
                    elif msg['role'] == 'user':
                        user_input = msg['content']
                    elif msg['role'] == 'assistant':
                        output = msg['content']
                
                items.append({
                    "instruction": instruction,
                    "input": user_input,
                    "output": output
                })
            elif 'conversations' in item:  # Альтернативный формат чата
                conversations = item['conversations']
                # Простая обработка: берем последнюю пару user-assistant
                for i in range(len(conversations) - 1):
                    if conversations[i]['from'] == 'human' and conversations[i+1]['from'] == 'gpt':
                        items.append({
                            "instruction": "",
                            "input": conversations[i]['value'],
                            "output": conversations[i+1]['value']
                        })
            else:
                # Стандартный формат
                items.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", "")
                })
        
        return prepare_dataset_for_chat(items, tokenizer, max_length, val_split)
        
    except Exception as e:
        print(f"Ошибка загрузки датасета {dataset_name}: {e}")
        raise

class SmartDataCollator:
    """Умный коллатор с динамическим паддингом и улучшенной обработкой батчей."""
    
    def __init__(self, tokenizer, max_length=512, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}
        
        # Динамический паддинг до максимальной длины в батче
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len, self.max_length)
        
        # Округляем до кратного pad_to_multiple_of
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch = {}
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        
        for key in ["input_ids", "attention_mask", "labels"]:
            if key not in features[0]:
                continue
                
            tensors = []
            for f in features:
                tensor = torch.tensor(f[key][:max_len], dtype=torch.long)
                
                # Паддинг
                pad_len = max_len - len(tensor)
                if pad_len > 0:
                    if key == "labels":
                        pad_value = -100
                    elif key == "attention_mask":
                        pad_value = 0
                    else:
                        pad_value = pad_token_id
                    
                    tensor = torch.nn.functional.pad(
                        tensor, 
                        (0, pad_len), 
                        value=pad_value
                    )
                
                tensors.append(tensor)
            
            if tensors:ёё
                batch[key] = torch.stack(tensors)
        
        return batch

# -------------------- Основная функция --------------------

def main():
    parser = argparse.ArgumentParser(description='Улучшенный fine-tuning causal LM с LoRA')
    
    # Основные параметры
    parser.add_argument('--model_name', type=str, default='gpt2', 
                       help='Название модели в HF')
    parser.add_argument('--dataset_name', type=str, required=False, 
                       help='Название датасета в HF')
    parser.add_argument('--data_path', type=str, 
                       help='Путь к data.json или data.jsonl')
    parser.add_argument('--output_dir', type=str, default='./fine-tuned-lora',
                       help='Директория для сохранения результатов')
    
    # Параметры обучения
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    
    # LoRA параметры
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default='',
                       help='Список модулей для LoRA через запятую')
    
    # Квантование
    parser.add_argument('--use_4bit', action='store_true',
                       help='Использовать 4-bit quantization')
    parser.add_argument('--use_8bit', action='store_true',
                       help='Использовать 8-bit quantization')
  
    
    args = parser.parse_args()
    
    # Настройка seed для воспроизводимости
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 60)
    print(f"Начало fine-tuning модели: {args.model_name}")
    print(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Загрузка токенизатора
    print("\n1. Загрузка токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            use_fast=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Ошибка загрузки токенизатора: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
# Проверяем, есть ли у токенизатора pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
           # Используем eos_token (токен конца последовательности) как pad_token
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Установлен pad_token как eos_token: {tokenizer.eos_token}")
      # Создаем новый pad_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print("Добавлен новый pad_token: [PAD]")
    
    # Загрузка модели
    print("\n2. Загрузка модели...")
    
    quantization_config = None
    torch_dtype = torch.float16
    
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16
        )
        torch_dtype = torch.float16
        print("Используется 4-bit quantization")
    elif args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        torch_dtype = torch.float16
        print("Используется 8-bit quantization")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto" if quantization_config else None,
            trust_remote_code=True,
            use_cache=not args.gradient_checkpointing
        )
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        # Пробуем загрузить без device_map для CPU
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    
    # Gradient checkpointing для экономии памяти
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Включен gradient checkpointing")
    
    # Подготовка модели для k-bit обучения
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Применение LoRA
    if args.use_lora:
        print("\n3. Настройка LoRA...")
        
        # Определяем целевые модули
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        else:
            target_modules = get_default_lora_targets(args.model_name)
        
        print(f"Целевые модули LoRA: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Загрузка данных
    print("\n4. Подготовка данных...")
    
    if args.data_path:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Файл {args.data_path} не найден")
        
        items = read_json_or_jsonl(args.data_path)
        if not items:
            raise ValueError("Данные пусты или не удалось прочитать файл")
        
        print(f"Загружено {len(items)} примеров из файла")
        
        # Конвертация различных форматов
        converted_items = []
        for item in items[:args.max_samples] if args.max_samples else items:
            # Различные форматы данных
            if 'prompt' in item and 'response' in item:
                converted_items.append({
                    "instruction": "",
                    "input": item.get('prompt', ''),
                    "output": item.get('response', '')
                })
            elif 'question' in item and 'answer' in item:
                converted_items.append({
                    "instruction": "",
                    "input": item.get('question', ''),
                    "output": item.get('answer', '')
                })
            elif 'text' in item:
                # Простой текст
                converted_items.append({
                    "instruction": "",
                    "input": "",
                    "output": item.get('text', '')
                })
            else:
                # Стандартный формат
                converted_items.append({
                    "instruction": item.get('instruction', ''),
                    "input": item.get('input', ''),
                    "output": item.get('output', '')
                })
        
        train_dataset, eval_dataset = prepare_dataset_for_chat(
            converted_items, 
            tokenizer, 
            args.max_length,
            args.val_split
        )
        
    elif args.dataset_name:
        train_dataset, eval_dataset = prepare_dataset_from_hf(
            args.dataset_name,
            tokenizer,
            max_samples=args.max_samples,
            max_length=args.max_length,
            val_split=args.val_split
        )
    else:
        raise ValueError("Должен быть указан либо --data_path, либо --dataset_name")
    
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError("Не удалось создать обучающий датасет или он пуст")
    
    print(f"Размер обучающего датасета: {len(train_dataset)}")
    if eval_dataset and len(eval_dataset) > 0:
        print(f"Размер валидационного датасета: {len(eval_dataset)}")
    else:
        print("Валидационный датасет не создан или пуст")
        eval_dataset = None
    
    # Коллатор данных
    data_collator = SmartDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=8
    )
    
    # Коллбэки
    callbacks = [
        GradientMonitorCallback(),
        LearningRateMonitorCallback(),
        TrainingProgressCallback()
    ]
    
    if args.early_stopping and eval_dataset and len(eval_dataset) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.01
            )
        )
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        
        # Параметры батча
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Параметры оптимизации
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        
        # Scheduler
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        
        # Сохранение и логирование
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch" if not args.early_stopping else "steps",
        save_steps=100 if args.early_stopping else None,
        save_total_limit=3,
        evaluation_strategy="epoch" if eval_dataset else "no",
        eval_steps=50 if eval_dataset and args.early_stopping else None,
        
        # Mixed precision
        fp16=torch.cuda.is_available() and not (args.use_4bit or args.use_8bit),
        bf16=False,
        
        # Оптимизатор
        optim="adamw_torch",
        
        # Отчетность
        report_to=["tensorboard"] if not (args.use_4bit or args.use_8bit) else [],
        logging_dir=os.path.join(args.output_dir, "logs"),
        
        # Другие параметры
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        dataloader_num_workers=min(4, os.cpu_count() or 1) if not (args.use_4bit or args.use_8bit) else 0,
        dataloader_pin_memory=True,
        gradient_checkpointing=args.gradient_checkpointing,

    )
    
    # Создание тренера
    print("\n5. Настройка тренера...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Обучение
    print("\n6. Начало обучения...")
    try:
        train_result = trainer.train()
        
        # Сохранение результатов
        print("\n7. Сохранение результатов...")
        
        # Сохраняем метрики
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        
        if eval_dataset:
            try:
                eval_metrics = trainer.evaluate()
                trainer.save_metrics("eval", eval_metrics)
            except Exception as e:
                print(f"Ошибка при оценке модели: {e}")
        
        # Сохраняем модель
        if args.use_lora:
            # Сохраняем только LoRA адаптер
            model.save_pretrained(args.output_dir)
            print(f"LoRA адаптер сохранен в {args.output_dir}")
        else:
            # Сохраняем полную модель
            model.save_pretrained(args.output_dir, safe_serialization=True)
            print(f"Полная модель сохранена в {args.output_dir}")
        
        # Сохраняем токенизатор
        tokenizer.save_pretrained(args.output_dir)
        
        # Сохраняем конфигурацию обучения
        config_file = os.path.join(args.output_dir, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("Обучение завершено успешно!")
        print(f"Результаты сохранены в: {args.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        print("Попытка сохранения промежуточных результатов...")
        
        try:
            # Пробуем сохранить модель, даже если обучение прервалось
            if args.use_lora:
                model.save_pretrained(args.output_dir + "_interrupted")
            else:
                model.save_pretrained(args.output_dir + "_interrupted", safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir + "_interrupted")
            print(f"Промежуточные результаты сохранены в: {args.output_dir}_interrupted")
        except Exception as save_error:
            print(f"Не удалось сохранить промежуточные результаты: {save_error}")
        
        raise

if __name__ == '__main__':
    main()