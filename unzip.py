import os
import zipfile


def unzip_all_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.zip'):
            # 開啟 ZIP 檔案並解壓縮到相同資料夾
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    file = file.encode('cp437').decode('gbk')
                    zip_ref.extract(file, folder_path)

folder_path = 'law/'
unzip_all_in_folder(folder_path)