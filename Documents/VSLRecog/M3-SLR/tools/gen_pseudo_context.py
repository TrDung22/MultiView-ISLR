import pandas as pd
import torch
from transformers import pipeline
import re

# Khởi tạo pipeline
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:1"
)

# Thiết lập pad_token cho tokenizer
pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

# Đọc file CSV đầu vào
input_csv = 'vsl/label1-200/testGloss.csv'
df = pd.read_csv(input_csv)
glosses = df['gloss'].tolist()

# Định nghĩa prompts
prompts = [
    "Create a story using {word} to highlight its semantic meaning. Answer using one sentence.",
    "Explain the meaning of {word} without mention it. Answer using one sentence.",
    "Just repeat the sentence 'This is {word}' without any other respone."
]

# Tạo danh sách messages
all_messages = []
for gloss in glosses:
    for prompt_template in prompts:
        message = [{"role": "user", "content": prompt_template.format(word=gloss)}]
        all_messages.append(message)

# Gọi pipeline để tạo văn bản
outputs = pipe(all_messages, max_new_tokens=256, batch_size=10)

def remove_assistant_prefix(text):
    # Loại bỏ khoảng trắng thừa ở đầu và cuối
    text = text.strip()
    # Dùng regex để loại bỏ tiền tố "assistant" ở đầu câu
    # Mẫu regex: ^assistant[:\s-]*\s*
    # - ^: bắt đầu chuỗi
    # - assistant: từ "assistant"
    # - [:\s-]*: bất kỳ dấu hai chấm, khoảng trắng, hoặc dấu gạch ngang nào
    # - \s*: bất kỳ số khoảng trắng nào sau đó
    pattern = r'^assistant[:\s-]*\s*'
    # Thay thế tiền tố bằng chuỗi rỗng, không phân biệt hoa/thường
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned_text

# Xử lý các văn bản được tạo ra để loại bỏ "assistant: "
generated_texts = [remove_assistant_prefix(output[0]["generated_text"][-1]["content"]) for output in outputs]

# Phân chia văn bản theo từng loại prompt
num_glosses = len(glosses)
story_texts = generated_texts[0::3]
explain_texts = generated_texts[1::3]
straight_texts = generated_texts[2::3]

# Gán vào DataFrame
df['story'] = story_texts
df['explain'] = explain_texts
df['straight'] = straight_texts

# Lưu kết quả vào file CSV
output_csv = 'vsl/label1-200/testLlama.csv'
df.to_csv(output_csv, index=False)
print(f"Đã lưu kết quả vào {output_csv}")