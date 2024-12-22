import os
import torch
#torch.cuda.set_device(2)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5" 

import json
import argparse
from sentence_transformers import SentenceTransformer, InputExample, util, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from torch.utils.data import DataLoader
from sentence_transformers.losses import ContrastiveLoss
import csv
from datasets import Dataset
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
import jieba
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(token="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")

# ==== 參數 ====
TRAIN_DATA_PATH = "train_data.jsonl"  # 訓練資料路徑
TEST_DATA_PATH = "test_data.jsonl"  # 測試資料路徑
REWRITE_TEST_DATA_PATH = "rewrite_test_data.jsonl"  # 測試資料路徑
DOCUMENTS_PATH = "document_pool.json"  # 法條資料路徑
MODEL_NAME = "BAAI/bge-m3"#"sentence-transformers/all-mpnet-base-v2"#"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"#
RERWITE_MODEL_NAME = "lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct"
OUTPUT_MODEL_PATH = "./retrieval_model"  # 訓練後模型儲存路徑
OUTPUT_FOLDER_PATH = "output/result.csv"
BATCH_SIZE = 16
EPOCHS = 1


import nltk
nltk.download('punkt_tab')

cache_dir = os.path.join(os.getcwd(), "jieba_cache")
os.makedirs(cache_dir, exist_ok=True)
jieba.tmp_dir = cache_dir

class Retriever:
    def __init__(self, model_name, rewrite_llm, rewriting=True,device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name).to(self.device)
        #self.model = torch.nn.DataParallel(self.model, device_ids=[2, 3])
        self.documents = None
        self.hybrid_alpha = 0.5
        """""
        if rewriting:
            self.rewrite_tokenizer = AutoTokenizer.from_pretrained(
                rewrite_llm, 
                device_map=self.device
            )

            self.rewrite_model = AutoModelForCausalLM.from_pretrained(rewrite_llm)
            self.rewrite_model.eval()
        """""
    def generate(self, messages, max_tokens):
        text_chat = self.rewrite_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.rewrite_tokenizer([text_chat], return_tensors="pt").to(self.rewrite_model.device)

        generated_ids = self.rewrite_model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            eos_token_id=self.rewrite_tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.rewrite_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    @staticmethod
    def strip_all_lines(s: str) -> str:
        """Remove all leading and trailing spaces of each line in the string."""
        return '\n'.join([line.strip() for line in s.splitlines()])
    @staticmethod
    def get_system_prompt() -> str:
        system_prompt = """\
        "你是一名專業的法律專家，精通台灣法律和相關法規。"
        "你的任務是根據用戶的問題，提供準確且詳盡的法律解釋，並引用相關法條進行佐證。回答時請保持專業且簡潔。""".strip()
        return Retriever.strip_all_lines(system_prompt)

    @staticmethod
    def get_zeroshot_prompt(
        text: str
    ) -> str:
        prompt = f"""\
        作為一位法律專家，回答下列問題:

        問題:
        {text}
        
        回答:""".strip()
        return Retriever.strip_all_lines(prompt)

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        {{question}}
        Diagnosis: {{answer}}"""
        return Retriever.strip_all_lines(prompt)

    @staticmethod
    def get_fewshot_template(
        shot: str,
        text: str,
    ) -> str:
        prompt = f"""\
        作為一位法律專家，回答下列問題:

        範例:
        {shot}

        問題:
        {text}
        
        回答:"""
        return Retriever.strip_all_lines(prompt)
    def rewrite(self, query, max_tokens):
        system_prompt = self.get_system_prompt()
        zeroshot_prompt = self.get_zeroshot_prompt(query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": zeroshot_prompt}
        ]

        message = self.generate(messages, max_tokens)
        return message
    def load_embedding_train_data(self, train_path, document_path):
        """讀取訓練資料 (jsonl 格式)"""
        sentence1 = []
        sentence2 = []
        label = []
        examples = []
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                query = data['question'] if data['question'] is not None else data['title']
                positives = data["label"].split(",")
                with open(document_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                    correspond_doc = []
                    for item in documents:
                        if item["label"] in positives:
                            correspond_doc.append(item["content"])
                #print(correspond_doc)
                for positive in correspond_doc:
                    sentence1.append(query)
                    sentence2.append(positive)
                    label.append(1)
                    examples.append(InputExample(texts=[query, positive], label=1.0))
        train_dataset = Dataset.from_dict({
            "sentence1": sentence1,
            "sentence2": sentence2,
            "label": label,
        })
        return train_dataset,examples
    def split_eval_train_data(self, train_path, eval_ratio=0.2):
        """讀取訓練資料 (jsonl 格式)"""
        sentence1 = []
        sentence2 = []

        data_list = [] 

        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    data_list.append(data)
                except json.JSONDecodeError:
                    continue

        random.shuffle(data_list)

        eval_size = int(len(data_list) * eval_ratio)
        eval_data = data_list[:eval_size]
        train_data = data_list[eval_size:]
    

        return eval_data, train_data
    def load_test_data(self, test_path):
        """讀取訓練資料 (jsonl 格式)"""
        data_list = [] 

        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    data_list.append(data)
                except json.JSONDecodeError:
                    continue
    

        return data_list
    def load_documents(self, doc_path):
        """讀取文件池 (json 格式)"""
        with open(doc_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
        self.documents =  {doc["label"]: doc["content"] for doc in documents}
    def bm_25(self, test_data_path, model_path, output_path, documents_path, evaluate=False, top_k=7):

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

        with open(output_path, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "TARGET"])

            for line in test_data_path:
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
    
    def hybrid(self, test_data, model_path, output_path, documents_path, evaluate=False, top_k=7):
        self.load_documents(documents_path)
        print("Loading model...")
        #self.model = SentenceTransformer(model_path).to(self.device)

        document_labels = list(self.documents.keys())
        document_contents = list(self.documents.values())

        print("Encoding documents...")
        tokenized_documents = [list(jieba.cut(doc)) for doc in document_contents]
        print("Encoding documents...")
        self.model.eval()
        doc_embeddings = self.model.encode(document_contents, convert_to_tensor=True)

        bm25 = BM25Okapi(tokenized_documents)
        bm25.k1 = 1.5  # 或 2.0
        bm25.b = 0.75
        print("Retrieving queries...")

        if evaluate:
            evaluate_score = 0
            for data in test_data:
                title = data.get("title", "")
                question = data.get("question", "")
                title = title if title is not None else ""
                question = question if question is not None else ""
                query = title + " " + question

                tokenized_query = list(jieba.cut(query))

                scores = bm25.get_scores(tokenized_query)
                top_k_indices = np.argsort(scores)[::-1][:top_k]
                top_labels = [document_labels[idx] for idx in top_k_indices]
                top_labels_string = ", ".join(top_labels)

                label = data.get("label", "")
                print(f"correct:{label}, guessed: {top_labels_string}")
                score = self.calculate_f1_score(top_labels_string, label)
                evaluate_score = evaluate_score+score

            print(f"evaluate_score_total:{evaluate_score}")
            evaluate_score = evaluate_score/len(test_data)
            print(f"Score: {evaluate_score}")
        else:
            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "TARGET"])

                for data in test_data:
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
    def rerank(self, test_data, model_path, output_path, documents_path, top_first_m=1000, top_k=3):
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

        
        with open(output_path, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "TARGET"])

            for data in test_data:
                query = data['question'] if data['question'] is not None else data['title']
                tokenized_query = list(jieba.cut(query))

                scores = bm25.get_scores(tokenized_query)

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
    def rewrite_bm25(self, test_data, model_path, output_path, documents_path, evaluate=False, top_k=7):
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

        if evaluate:
            evaluate_score = 0
            for data in test_data:
                query = data.get("context", "")

                tokenized_query = list(jieba.cut(query))

                scores = bm25.get_scores(tokenized_query)
                top_k_indices = np.argsort(scores)[::-1][:top_k]
                top_labels = [document_labels[idx] for idx in top_k_indices]
                top_labels_string = ", ".join(top_labels)

                label = data.get("label", "")
                print(f"correct:{label}, guessed: {top_labels_string}")
                score = self.calculate_f1_score(top_labels_string, label)
                evaluate_score = evaluate_score+score

            print(f"evaluate_score_total:{evaluate_score}")
            evaluate_score = evaluate_score/len(test_data)
            print(f"Score: {evaluate_score}")
        else:
            with open(output_path, mode="w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["id", "TARGET"])

                for data in test_data:
                    query = data.get("context", "")

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

    def calculate_f1_score(self, output, correct_output):
        output_set = set(output.split(","))
        correct_set = set(correct_output.split(","))

        tp = len(output_set & correct_set)
        fp = len(output_set - correct_set)
        fn = len(correct_set - output_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1
    def train(self, train_data_path, output_model_path, document_path, epochs=1, batch_size=32):
        """訓練檢索模型"""
        print("Loading training data...")
        train_data, old_train_data = self.load_embedding_train_data(train_data_path, document_path)
        #train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = ContrastiveLoss(self.model)
        """""
        train_dataloader = DataLoader(old_train_data, shuffle=True, batch_size=batch_size)

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
            report_to="none" 
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            train_dataset=train_data,
            loss=train_loss,
            args=args
        )
        trainer.train()
        
        print(f"Model saved to {output_model_path}")
    
    def retrieve(self, test_data, model_path, output_path, documents_path, evaluate=False, top_k=3):
        self.load_documents(documents_path)
        print("Loading model...")
        #self.model = SentenceTransformer(model_path).to(self.device)

        document_labels = list(self.documents.keys())
        document_contents = list(self.documents.values())

        print("Encoding documents...")
        doc_embeddings = self.model.encode(document_contents, convert_to_tensor=True)

        print("Retrieving queries...")

        
        with open(output_path, mode="w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "TARGET"])

            for data in test_data:
                query = [data['question'] if data['question'] is not None else data['title']]

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
    parser.add_argument("--rewrite_test_data", type=str, default=REWRITE_TEST_DATA_PATH, help="重寫測試資料路徑")
    parser.add_argument("--documents", type=str, default=DOCUMENTS_PATH, help="法律文件資料路徑")
    parser.add_argument("--output_model", type=str, default=OUTPUT_MODEL_PATH, help="儲存訓練後的模型路徑")

    parser.add_argument("--model", type=str, default=MODEL_NAME, help="模型")
    parser.add_argument("--rewrite_model", type=str, default=RERWITE_MODEL_NAME, help="模型")
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER_PATH, help="儲存結果")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="訓練回合數")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="訓練批次大小")

    args = parser.parse_args()
    #torch.cuda.empty_cache()
    retriever = Retriever(model_name=args.model, rewrite_llm=args.rewrite_model, rewriting=False, device="cuda:1")

    """""
    with open(args.test_data, "r", encoding="utf-8") as f:
            with open(args.rewrite_test_data, mode="w", encoding="utf-8") as file:
                for index, line in enumerate(f):
                    print("line; "+ str(index))
                    data = json.loads(line.strip())
                    title = data.get("title", "")
                    question = data.get("question", "")
                    title = title if title is not None else ""
                    question = question if question is not None else ""
                    query = title + " " + question
                    
                    rewrite_query = retriever.rewrite(query, 128)
                    print(f"Query: {rewrite_query}")
                    new_data = {
                        "id": data.get("id", ""),
                        "context": rewrite_query
                    }
                    file.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                print("Finish!")
    """""

    #eval_data, train_data = retriever.split_eval_train_data(args.train_data)
    #test_data = retriever.load_test_data(args.rewrite_test_data)

    retriever.train(train_data_path=args.train_data , output_model_path=args.output_model, document_path=args.documents, epochs=args.epochs, batch_size=args.batch_size)
    #retriever.retrieve(args.test_data, args.output_model, args.output_folder, args.documents)
    #retriever.rewrite_bm25(eval_data, args.output_model, args.output_folder, args.documents, evaluate=True)
    data_list = [] 

    with open(args.test_data, 'r', encoding='utf-8') as file1, open(args.rewrite_test_data, 'r', encoding='utf-8') as file2:
        for line1, line2 in zip(file1, file2):
            try:
                data1 = json.loads(line1.strip())
                data2 = json.loads(line2.strip())

                data1["question"] = (data1["question"] if data1["question"] is not None else "") + "\n" + (data2["context"] if data2["context"] is not None else "")
                data_list.append(data1)
            except json.JSONDecodeError:
                continue
    print("\n".join([data["question"] for data in data_list if "question" in data]))

    
    retriever.retrieve(data_list, args.output_model, args.output_folder, args.documents, evaluate=False)
    #retriever.rerank(data_list, args.output_model, args.output_folder, args.documents)
    

if __name__ == "__main__":
    main()