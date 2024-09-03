# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
import faiss
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Constants and configurations
MODEL_NAME = "meta-llama/llama-7b" 
TRAIN_EPOCHS = 3
SCRAPE_URL = "https://www.mehuljaiswal.com" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Scrape the website
def scrape_website(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    paragraphs = soup.find_all("p")
    texts = [p.get_text() for p in paragraphs]
    headers = soup.find_all(["h1", "h2", "h3"])
    texts += [header.get_text() for header in headers]
    return " ".join(texts)


def build_index(documents):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    
    index = faiss.IndexFlatL2(X.shape[1])
    faiss.normalize_L2(X.toarray())
    index.add(X.toarray())
    
    return index, vectorizer

# Step 3: Fine-tune LLaMA on the scraped content
def fine_tune_llama(model, tokenizer, documents):
    inputs = tokenizer(documents, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs
    )
    trainer.train()

def answer_question(query, index, vectorizer, model, tokenizer):
    query_vector = vectorizer.transform([query])
    faiss.normalize_L2(query_vector.toarray())
    
    D, I = index.search(query_vector.toarray(), 1)  # Get the closest document
    
    context = vectorizer.inverse_transform(index.reconstruct(int(I[0])))[0]
    input_text = f"{context}\n\nQuestion: {query}\nAnswer:"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    
    output = model.generate(inputs, max_length=150)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer

def main():
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    
    document = scrape_website(SCRAPE_URL)
    
    fine_tune_llama(model, tokenizer, document)
    
    index, vectorizer = build_index([document])
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = answer_question(query, index, vectorizer, model, tokenizer)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main()
