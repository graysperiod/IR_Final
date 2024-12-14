import json
import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


# ==== 參數 ====
TRAIN_DATA_PATH = "train_data.jsonl"  # 訓練資料路徑
TEST_DATA_PATH = "test_data.jsonl"  # 測試資料路徑
DOCUMENTS_PATH = "document_pool.json"  # 法條資料路徑
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_MODEL_PATH = "./retrieval_model"  # 訓練後模型儲存路徑
OUTPUT_FOLDER_PATH = "output/"
BATCH_SIZE = 16
EPOCHS = 1

# ==== 載入資料 ====
def load_train_data(train_path):
    """讀取訓練資料 (jsonl 格式)"""
    examples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            title = data["title"]
            query = data["question"]
            positives = data["label"].split(",")
            for positive in positives:
                examples.append(InputExample(texts=[title+" "+query, positive], label=1.0))
    return examples

def load_documents(doc_path):
    """讀取文件池 (json 格式)"""
    with open(doc_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    return {doc["label"]: doc["content"] for doc in documents}

def load_test_data(test_data_path):
    """讀取測試資料 (jsonl 格式)"""
    test_queries = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            test_queries.append(data["title"]+" "+data["question"])
    return test_queries

# ==== 訓練 ====
def train_retrieval_model(train_data_path, output_model_path, model, epochs, batch):
    # 載入模型與資料
    print("Training")
    train_data = load_train_data(train_data_path)
    print("Finish loading data")
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch)
    train_loss = losses.CosineSimilarityLoss(model)

    # 訓練模型
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_model_path
    )
    """""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training", unit="batch")):
            model.zero_grad()
            loss_value = train_loss(batch)
            loss_value.backward()

            # 每幾個步驟顯示一下訓練損失
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss_value.item():.4f}")
    """""

    print(f"模型已儲存至 {output_model_path}")

# ==== 推理 ====
def retrieve(queries, model_path, documents):
    """檢索邏輯"""
    model = SentenceTransformer(model_path)
    document_labels = list(documents.keys())
    document_contents = list(documents.values())

    doc_embeddings = model.encode(document_contents, convert_to_tensor=True)

    for query in queries:
        print(f"查詢: {query}")

        query_embedding = model.encode(query, convert_to_tensor=True)
        
        cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        top_results = torch.topk(cos_scores, k=5)  # 取前 5 個結果
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                "label": document_labels[idx],
                "content": document_contents[idx],
                "score": score.item()
            })

        print("檢索結果：")
        for result in results:
            print(f"[{result['label']}] (Score: {result['score']:.4f})")
            print(result["content"])
            print()

# ==== 主程式 ====
def main():
    parser = argparse.ArgumentParser(description="訓練並檢索法律文件相關資料")
    parser.add_argument("--train_data", type=str, default=TRAIN_DATA_PATH, help="訓練資料路徑")
    parser.add_argument("--test_data", type=str, default=TEST_DATA_PATH, help="測試資料路徑")
    parser.add_argument("--documents", type=str, default=DOCUMENTS_PATH, help="法律文件資料路徑")
    parser.add_argument("--output_model", type=str, default=OUTPUT_MODEL_PATH, help="儲存訓練後的模型路徑")

    parser.add_argument("--model", type=str, default=MODEL_NAME, help="模型")
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER_PATH, help="儲存結果")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="訓練回合數")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="訓練批次大小")

    args = parser.parse_args()

    # Step 1: 訓練模型
    model = SentenceTransformer(args.model)
    device = torch.device("cuda 0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = model.to(device)

    train_retrieval_model(args.train_data, args.output_model, model, args.epochs, args.batch_size)

    # Step 2: 推理測試
    documents = load_documents(args.documents)
    test_queries = load_test_data(args.test_data)

    results = retrieve(test_queries, args.output_model, documents)

    # 輸出檢索結果
    print("檢索結果：")
    for result in results:
        print(f"[{result['label']}] (Score: {result['score']:.4f})")
        print(result["content"])
        print()

if __name__ == "__main__":
    main()