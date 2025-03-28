import torch
from transformers import pipeline

# Khởi tạo pipeline
model_id = "google/gemma-3-4b-it"
pipe = pipeline(
    "text2text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:1"
)

# Chuỗi chứa các từ gloss
gloss_str = "run fast, run slow, run, walk, balk, grass, grandfather, sing, car, watch"

# Chuyển chuỗi thành danh sách các từ
gloss_list = [word.strip() for word in gloss_str.split(',')]

# Tạo danh sách các prompts (chuỗi văn bản) thay vì messages_list
prompts = [
    f"Create a story using {word} to highlight its semantic meaning. Answer using one sentence."
    for word in gloss_list
]

# Gọi pipeline với danh sách prompts
outputs = pipe(prompts, max_new_tokens=256)

# Lấy model và tokenizer từ pipeline
model = pipe.model
tokenizer = pipe.tokenizer

# Xử lý kết quả và lấy embedding
for word, output in zip(gloss_list, outputs):
    # Trích xuất generated text
    generated_text = output["generated_text"]
    print(f"Gloss '{word}':")
    print(generated_text)
    print("-" * 40)
    
    # Tokenize generated text
    input_ids = tokenizer(generated_text, return_tensors="pt").input_ids.to("cuda:1")
    
    # Lấy hidden states
    with torch.no_grad():
        model_output = model(input_ids, output_hidden_states=True)
    
    # Lấy embedding
    hidden_states = model_output.hidden_states[-1]  # Layer cuối
    embedding = torch.mean(hidden_states[0], dim=0)
    
    # In hoặc lưu embedding nếu cần
    print(f"Embedding shape: {embedding.shape}")