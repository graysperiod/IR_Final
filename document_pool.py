import json
import re
import os
from striprtf.striprtf import rtf_to_text

def turn_json(folder_path, output_file):
    document_pool = []

    for filename in os.listdir(folder_path):
        input_path = os.path.join(folder_path, filename)

        #with open(input_path, 'r') as file:
        #    input_file = file.read()
        print(input_path)
        with open(input_path) as infile:
            content = infile.read()
            input_file = rtf_to_text(content)
        #print(input_file)

        patterns = [
            r"第\s*[一二三四五六七八九十百零\d]+\s*章\s*.*",  # 移除「第 幾 章」的行
            r"第\s*[一二三四五六七八九十百零\d]+\s*節\s*.*",  # 移除「第 幾 節」的行
        ]
        for pattern in patterns:
            input_file = re.sub(pattern, '', input_file)

        text = re.sub(r'\n+', '\n', input_file).strip()

        law_name_match = re.search(r"法規名稱：\s*(.+)", input_file)
        if law_name_match:
            law_name = law_name_match.group(1).strip()
        else:
            raise ValueError("未找到法規名稱！請確認文本格式。")

        pattern = r"第\s*(\d+(-\d+)?)\s*條\n(.*?)\n(?=(第\s*\d+(-\d+)?\s*條|$))"
        matches = re.finditer(pattern, input_file, re.DOTALL)

        for match in matches:
            article_number = match.group(1)
            content = match.group(3).strip()

            # 將條號中的 "-" 替換為 "條之"
            article_number = article_number.replace("-", "條之")

            # 建立 label
            label = f"{law_name}第{article_number}條".strip()
            document_pool.append({"label": label, "content": content})

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(document_pool, outfile, ensure_ascii=False, indent=4)

output_file = "document_pool.json"
folder_path = 'law_unzip/'
turn_json(folder_path, output_file)