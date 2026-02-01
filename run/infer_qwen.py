import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = os.environ["QWEN_MODEL_DIR"]

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

prompt = "用中文回答：解释一下什么是NSCC的PBS作业调度，并给一个qsub例子。"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)

print(tok.decode(out[0], skip_special_tokens=True))