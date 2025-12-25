from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Проверяем доступность CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Загружаем модели
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,
    device_map="auto"
)

try:
    lora_model = PeftModel.from_pretrained(base_model, "./my-lora-model")
    print("LoRA модель успешно загружена")
except Exception as e:
    print(f"Ошибка при загрузке LoRA модели: {e}")
    lora_model = base_model

# Перемещаем модели на нужное устройство
base_model = base_model.to(device)
lora_model = lora_model.to(device)

# Сообщения для тестирования
messages_list = [
    [{"role": "user", "content": "What are the main differences between Crohn's disease and ulcerative colitis in terms of symptoms, treatment, and lifestyle impact?"}],
    [{"role": "user", "content": "I have running nose and sore throat, what can it be?"}],
    [{"role": "user", "content": "Hello, I have bad and painful acne on face and body. How can I get rid of it?"}]
]

for i, messages in enumerate(messages_list):
    print(f"\n{'='*80}")
    print(f"Тест {i+1}: {messages[0]['content'][:50]}...")
    print('='*80)

    # Формируем промпт для Qwen
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        # Fallback если apply_chat_template не работает
        user_message = messages[0]['content']
        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    # Токенизируем
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    try:
        # Генерация с базовой моделью
        with torch.no_grad():
            outputs_base = base_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # Генерация с LoRA-моделью
        with torch.no_grad():
            outputs_lora = lora_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        # Декодируем ответы
        text_base = tokenizer.decode(outputs_base[0], skip_special_tokens=True)
        text_lora = tokenizer.decode(outputs_lora[0], skip_special_tokens=True)

        # Извлекаем только ответ ассистента
        if prompt in text_base:
            base_response = text_base.split(prompt)[-1].strip()
        else:
            base_response = text_base

        if prompt in text_lora:
            lora_response = text_lora.split(prompt)[-1].strip()
        else:
            lora_response = text_lora

        print(f"\nВопрос: {messages[0]['content']}")
        print(f"\n{'='*40} Базовая модель {'='*40}")
        print(base_response)
        print(f"\n{'='*40} LoRA модель {'='*43}")
        print(lora_response)

    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        continue

print(f"\n{'='*80}")
print("Тестирование завершено!")