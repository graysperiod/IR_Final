import json
import argparse
from sentence_transformers import SentenceTransformer, InputExample, util, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import ContrastiveLoss
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import csv
import os
from datasets import Dataset
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
import jieba

# ==== 參數 ====
TRAIN_DATA_PATH = "IR_Final/train_data.jsonl"  # 訓練資料路徑
TEST_DATA_PATH = "IR_Final/test_data.jsonl"  # 測試資料路徑
DOCUMENTS_PATH = "IR_Final/document_pool.json"  # 法條資料路徑
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"#"sentence-transformers/all-mpnet-base-v2"
OUTPUT_MODEL_PATH = "./IR_Final/retrieval_model"  # 訓練後模型儲存路徑
OUTPUT_FOLDER_PATH = "IR_Final/output/result.csv"
BATCH_SIZE = 16
EPOCHS = 1

#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import nltk
nltk.download('punkt_tab')

class Retriever:
    def __init__(self, model_name, device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)

        self.documents = None
        self.hybrid_alpha = 0.5
    def load_train_data(self, train_path, document_path):
        """讀取訓練資料 (jsonl 格式)"""
        sentence1 = []
        sentence2 = []
        label = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                title = data.get("title", "")
                question = data.get("question", "")
                title = title if title is not None else ""
                question = question if question is not None else ""
                positives = data["label"].split(",")
                with open(document_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                    correspond_doc = []
                    for item in documents:
                        if item["label"] in positives:
                            correspond_doc.append(item["content"])
                #print(correspond_doc)
                for positive in correspond_doc:
                    sentence1.append(title + " " + question)
                    sentence2.append(positive)
                    label.append(1)
                    #examples.append(InputExample(texts=[title + " " + question, positive], label=1.0))
        train_dataset = Dataset.from_dict({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": label,
        })
        return train_dataset

    def load_documents(self, doc_path):
        """讀取文件池 (json 格式)"""
        with open(doc_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        self.documents =  {doc["label"]: doc["content"] for doc in documents}
    def bm_25(self, test_data_path, model_path, output_path, documents_path, top_k=7):

        self.load_documents(documents_path)
        print("Loading model...")
        #self.model = SentenceTransformer(model_path).to(self.device)

        document_labels = list(self.documents.keys())
        document_contents = list(self.documents.values())

        print("Encoding documents...")
        tokenized_documents = [list(jieba.cut(doc)) for doc in document_contents]
        bm25 = BM25Okapi(tokenized_documents)
        bm25.k1 = 1.5  # 或 2.0
        bm25.b = 0.75
        print("Retrieving queries...")
        with open(test_data_path, "r", encoding="utf-8") as f:
            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "TARGET"])

                for line in f:
                    data = json.loads(line.strip())
                    title = data.get("title", "")
                    question = data.get("question", "")
                    title = title if title is not None else ""
                    question = question if question is not None else ""
                    query = title + " " + question

                    #print(f"Query: {query}")
                    tokenized_query = list(jieba.cut(query))

                    scores = bm25.get_scores(tokenized_query)
                    #top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
                    #print("最相關文檔索引：", [doc[0] for doc in top_docs])
                    top_k_indices = np.argsort(scores)[::-1][:top_k]
                    #print(top_k_indices)
                    top_labels = [document_labels[idx] for idx in top_k_indices]
                    top_labels_string = ", ".join(top_labels)
                    #print(f"Top labels: {top_labels}")
                    writer.writerow([data["id"], top_labels_string])
                print("Finish")
    def hybrid(self, test_data_path, model_path, output_path, documents_path, top_k=7):
        self.load_documents(documents_path)
        print("Loading model...")
        #self.model = SentenceTransformer(model_path).to(self.device)

        document_labels = list(self.documents.keys())
        document_contents = list(self.documents.values())

        print("Encoding documents...")
        tokenized_documents = [list(jieba.cut(doc)) for doc in document_contents]
        print("Encoding documents...")
        doc_embeddings = self.model.encode(document_contents, convert_to_tensor=True)

        bm25 = BM25Okapi(tokenized_documents)
        bm25.k1 = 1.5  # 或 2.0
        bm25.b = 0.75
        print("Retrieving queries...")
        with open(test_data_path, "r", encoding="utf-8") as f:
            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "TARGET"])

                for line in f:
                    data = json.loads(line.strip())
                    title = data.get("title", "")
                    question = data.get("question", "")
                    title = title if title is not None else ""
                    question = question if question is not None else ""
                    query = title + " " + question

                    #print(f"Query: {query}")
                    tokenized_query = list(jieba.cut(query))
                    bm25_scores = bm25.get_scores(tokenized_query)
                    
                    query_embedding = self.model.encode(query, convert_to_tensor=True)
                    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

                    bm25_scores_normalized = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

                    cos_scores_normalized = (cos_scores + 1) / 2

                    bm25_weight = 0.5
                    embedding_weight = 0.5

                    # 混合兩種方法的分數
                    final_scores = bm25_weight * np.array(bm25_scores_normalized) + embedding_weight * cos_scores_normalized.cpu().numpy()

                    top_k_indices = np.argsort(final_scores)[::-1][:top_k]
                    #print(top_k_indices)
                    top_labels = [document_labels[idx] for idx in top_k_indices]
                    top_labels_string = ", ".join(top_labels)
                    #print(f"Top labels: {top_labels}")
                    writer.writerow([data["id"], top_labels_string])
                print("Finish")
    def rerank(self, test_data_path, model_path, output_path, documents_path, top_first_m=1000, top_k=5):
        self.load_documents(documents_path)
        print("Loading model...")
        #self.model = SentenceTransformer(model_path).to(self.device)

        document_labels = list(self.documents.keys())
        document_contents = list(self.documents.values())

        print("Encoding documents...")
        tokenized_documents = [list(jieba.cut(doc)) for doc in document_contents]
        bm25 = BM25Okapi(tokenized_documents)
        bm25.k1 = 1.5  # 或 2.0
        bm25.b = 0.75
        print("Encoding documents...")
        doc_embeddings = self.model.encode(document_contents, convert_to_tensor=True)
        
        with open(test_data_path, "r", encoding="utf-8") as f:
            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "TARGET"])

                for line in f:
                    #print("Retrieving queries...")
                    data = json.loads(line.strip())
                    title = data.get("title", "")
                    question = data.get("question", "")
                    title = title if title is not None else ""
                    question = question if question is not None else ""
                    query = title + " " + question

                    #print(f"Query: {query}")
                    tokenized_query = list(jieba.cut(query))

                    scores = bm25.get_scores(tokenized_query)
                    #top_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
                    #print("最相關文檔索引：", [doc[0] for doc in top_docs])
                    top_k_indices = np.argsort(scores)[::-1][:top_first_m]
                    top_k_indices = np.argsort(top_k_indices)
                    #print(top_k_indices)
                    first_retrieve_embeddings = doc_embeddings[top_k_indices]
                    first_retrieve_labels = [document_labels[idx] for idx in top_k_indices]

                    query_embedding = self.model.encode(query, convert_to_tensor=True)
                    cos_scores = util.pytorch_cos_sim(query_embedding, first_retrieve_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=top_k)
                    #print(top_results)
                    top_labels = [first_retrieve_labels[idx] for idx in top_results.indices]
                    top_labels_string = ", ".join(top_labels)
                    #print(f"Top labels: {top_labels}")
                    writer.writerow([data["id"], top_labels_string])
                print("Finish")
    def train(self, train_data_path, output_model_path, document_path, epochs=1, batch_size=32):
        """訓練檢索模型"""
        print("Loading training data...")
        train_data = self.load_train_data(train_data_path, document_path)
        #train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = ContrastiveLoss(self.model)

        print("Training model...")
        print(self.model.device)
        """""
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_model_path
        )
        """""
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir=output_model_path,
            # Optional training parameters:
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            #per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            save_strategy="no", 
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            train_dataset=train_data,
            loss=train_loss,
            args=args
        )
        trainer.train()
        print(f"Model saved to {output_model_path}")

    def retrieve(self, test_data_path, model_path, output_path, documents_path, top_k=5):
        self.load_documents(documents_path)
        print("Loading model...")
        #self.model = SentenceTransformer(model_path).to(self.device)

        document_labels = list(self.documents.keys())
        document_contents = list(self.documents.values())

        print("Encoding documents...")
        doc_embeddings = self.model.encode(document_contents, convert_to_tensor=True)

        print("Retrieving queries...")
        with open(test_data_path, "r", encoding="utf-8") as f:
            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "TARGET"])

                for line in f:
                    data = json.loads(line.strip())
                    title = data.get("title", "")
                    question = data.get("question", "")
                    title = title if title is not None else ""
                    question = question if question is not None else ""
                    query = title + " " + question

                    #print(f"Query: {query}")
                    query_embedding = self.model.encode(query, convert_to_tensor=True)

                    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=top_k)
                    #print(top_results)
                    top_labels = [document_labels[idx] for idx in top_results.indices]
                    top_labels_string = ", ".join(top_labels)
                    #print(f"Top labels: {top_labels}")
                    writer.writerow([data["id"], top_labels_string])
                print("Finish!")



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
    #torch.cuda.empty_cache()
    retriever = Retriever(model_name=args.model, device="cuda:0")
    #retriever = Retriever(model_name='IR_Final/retrieval_model/checkpoint-4700', device="cuda:0")
    retriever.train(train_data_path=args.train_data , output_model_path=args.output_model, document_path=args.documents, epochs=args.epochs, batch_size=args.batch_size)
    #retriever.retrieve(args.test_data, args.output_model, args.output_folder, args.documents)
    #retriever.bm_25(args.test_data, args.output_model, args.output_folder, args.documents)
    retriever.rerank(args.test_data, args.output_model, args.output_folder, args.documents)
    

if __name__ == "__main__":
    main()