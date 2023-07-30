import csv
import json
import os


files_path = ['train_first_embed.json', 'train_second_embed.json', 'train_anomalies_embed.json']

for i in range(len(files_path)):
    data_path = os.path.join('Training_Data', files_path[i])
    with open(data_path, 'r', encoding='utf-8') as first_file:
        all_data = first_file.readlines()
        if i == 0:
            csv_text_path = os.path.join('data', 'first_text.csv')
            csv_triple_path = os.path.join('data', 'first_triple.csv')
        elif i == 1:
            csv_text_path = os.path.join('data', 'second_text.csv')
            csv_triple_path = os.path.join('data', 'second_triple.csv')
        elif i == 2:
            csv_text_path = os.path.join('data', 'anomalies_text.csv')
            csv_triple_path = os.path.join('data', 'anomalies_triple.csv')

        with open(csv_text_path, 'w', encoding='utf-8', newline='') as f1:
            writer1 = csv.writer(f1)
            writer1.writerows([["id", "sentiment", "review"]])
            data = list()
            for line in all_data:
                item = json.loads(line)
                order_number = str(item["Order_number"])
                text = item["text"]
                if isinstance(text, list):
                    text = ''.join(text)
                text = text.replace('\n', '')
                T = [order_number, '1', text]
                data.append(T)
            writer1.writerows(data)
            f1.close()


        with open(csv_triple_path, 'w', encoding='utf-8', newline='') as f2:
            writer2 = csv.writer(f2)
            writer2.writerows([["id", "sentiment", "review"]])
            data = list()
            for line in all_data:
                item = json.loads(line)
                order_number = str(item["Order_number"])
                triplet = str(item["Triplet"])
                event_type = item["event_type"]
                T = [order_number, '1', triplet]
                data.append(T)
            writer2.writerows(data)




