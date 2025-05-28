from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

MODEL_PATH = "content/chatbot_gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)  # float16 hatası olursa float32 yap

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["prompt"]
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=80)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
