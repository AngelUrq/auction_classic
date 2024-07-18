import os
import torch

folders = os.listdir('data')
output_folder = 'output'

os.makedirs(output_folder, exist_ok=True)

for folder in folders:
    subfolders = os.listdir('data/' + folder)

    for subfolder in subfolders:
        files = os.listdir('data/' + folder + '/' + subfolder)
        print('Processing folder:', folder + '/' + subfolder)

        auctions = {}

        for file in files:
            data = torch.load('data/' + folder + '/' + subfolder + '/' + file)

            item_id = file.split('.')[0]

            auctions[item_id] = data

        os.makedirs(output_folder + '/' + folder, exist_ok=True)
        torch.save(auctions, output_folder + '/' + folder + '/' + subfolder + '.pt')
        