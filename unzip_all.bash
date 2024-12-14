for zip_file in law/*.zip; do
  unzip -O gbk  "$zip_file" -d "law_unzip/"
done