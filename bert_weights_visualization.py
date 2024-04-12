# from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model
# import numpy as np
#
# # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2-large')
# # text = "Replace me by any text you'd like."
# # encoded_input = tokenizer(text, return_tensors='pt')
# # output = model(**encoded_input)
# np.savez('gpt2_large_weights.npz', **model.state_dict())
#
# # # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # model = BertModel.from_pretrained("bert-large-uncased")
# # # text = "Replace me by any text you'd like."
# # # encoded_input = tokenizer(text, return_tensors='pt')
# # # output = model(**encoded_input)
# # np.savez('bert_large_uncased_weights.npz', **model.state_dict())


# ######################################
# # QQT, VProj, W1W2 (GPT2)
# ######################################
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn as nn
#
# cols = 2
# rows = 2
# fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(25, 7))
# axes = axes.flat
# gpt2_weights = np.load('gpt2_weights.npz')
# for k, v in gpt2_weights.items():
#     print(k, v.shape)
#
# # creating a colormap
# colormap = sns.color_palette("Greys_r")
#
# for i in range(0, 4):
#     # query_weights = gpt2_weights[f'encoder.layer.{i}.attention.self.query.weight']
#     # key_weights = gpt2_weights[f'encoder.layer.{i}.attention.self.key.weight']
#     # value_weights = gpt2_weights[f'encoder.layer.{i}.attention.self.value.weight']
#     # proj_weights = gpt2_weights[f'encoder.layer.{i}.attention.output.dense.weight']
#     fc1 = gpt2_weights[f'h.{i+8}.mlp.c_fc.weight']  # [768, 3072]
#     fc2 = gpt2_weights[f'h.{i+8}.mlp.c_proj.weight'].T  # [768, 3072]
#     # QKT = np.absolute(np.sqrt(768) * np.matmul(query_weights, key_weights.T))
#     # VProj = np.absolute(np.sqrt(768) * np.matmul(value_weights, proj_weights))
#     W1W2 = np.absolute(np.sqrt(768) * np.matmul(fc1, fc2.T))
#
#     sns.heatmap(W1W2, ax=axes[i], cmap=colormap)
#     axes[i].set_title(f'layer - {i}')
# plt.show()







#####################################
# QQT, VProj, W1W2
#####################################
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

cols = 4
rows = 3
fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(25, 7))
axes = axes.flat
bert_base_weights = np.load('bert_large_uncased_weights.npz')
for k, v in bert_base_weights.items():
    print(k, v.shape)

# creating a colormap
colormap = sns.color_palette("Greys_r")

for i in range(12):
    query_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.query.weight']
    key_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.key.weight']
    value_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.value.weight']
    proj_weights = bert_base_weights[f'encoder.layer.{i}.attention.output.dense.weight']
    fc1 = bert_base_weights[f'encoder.layer.{i}.intermediate.dense.weight'].T  # [768, 3072]
    fc2 = bert_base_weights[f'encoder.layer.{i}.output.dense.weight']  # [768, 3072]
    QKT = np.absolute(np.sqrt(768) * np.matmul(query_weights, key_weights.T))
    VProj = np.absolute(np.sqrt(768) * np.matmul(value_weights, proj_weights))
    W1W2 = np.absolute(np.sqrt(768) * np.matmul(fc1, fc2.T))

    # query_head0 = query_weights.reshape((-1, 12, 64))[:, 11, :]
    # key_head0 = key_weights.reshape((-1, 12, 64))[:, 11, :]
    # QKT_head = np.absolute(np.sqrt(768) * np.matmul(query_head0, key_head0.T))
    # QQT_head = np.absolute(np.sqrt(768) * np.matmul(query_head0, query_head0.T))[:32, :32]

    sns.heatmap(W1W2, ax=axes[i], cmap=colormap)
    axes[i].set_title(f'layer - {i}')
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# colormap = sns.color_palette("Greys_r")
# bert_base_weights = np.load('bert_base_uncased_weights.npz')
# fc1 = bert_base_weights[f'encoder.layer.0.intermediate.dense.weight'].T  # [768, 3072]
# fc2 = bert_base_weights[f'encoder.layer.0.output.dense.weight']  # [768, 3072]
# W1W2 = (np.sqrt(768) * np.matmul(fc2.T, fc2))
# print(W1W2[:, 306])
# print(W1W2[:, 307])
# print(W1W2[:, 308])
# print(W1W2[:, 309])
# print(W1W2[:, 310])
# data1 = [W1W2[:, 306], W1W2[:, 307], W1W2[:, 308], W1W2[:, 309], W1W2[:, 310]]
# data2 = [fc2.T[:, 306], fc2.T[:, 307], fc2.T[:, 308], fc2.T[:, 309], fc2.T[:, 310]]
# # data3 = [fc2.T[:, 174], fc2.T[:, 175], fc2.T[:, 176], fc2.T[:, 177], fc2.T[:, 178]]
# # layer0_query_head0 = bert_base_weights['encoder.layer.10.attention.self.query.weight'].reshape((-1, 12, 64))[:, 0, :]
# # layer0_QQT = np.absolute(np.sqrt(768) * np.matmul(layer0_query_head0, layer0_query_head0.T))
# sns.heatmap(W1W2, cmap=colormap)
# fig, ax = plt.subplots()
# # ax.boxplot(data1, showfliers=False)
# # ax.boxplot(data2, showfliers=False)
# plt.show()




# # ######################################
# # # box plot
# # ######################################
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
#
# bert_base_weights = np.load('bert_large_uncased_weights.npz')
# fig, ax = plt.subplots()
#
# data = []
# cos_sim = []
# for i in range(24):
#     query_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.query.weight']
#     key_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.key.weight']
#     value_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.value.weight']
#     proj_weights = bert_base_weights[f'encoder.layer.{i}.attention.output.dense.weight']
#     fc1 = bert_base_weights[f'encoder.layer.{i}.intermediate.dense.weight']  # [3072, 768]
#     fc2 = bert_base_weights[f'encoder.layer.{i}.output.dense.weight'].T  # [3072, 768]
#     # data += [query_weights.reshape(-1)[:50]]
#     data += [fc2.reshape(-1)]
#
#     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#     # output = cos(torch.FloatTensor(fc1), torch.FloatTensor(fc2))
#     output = cos(torch.FloatTensor(fc1).T, torch.FloatTensor(fc2).T)  # [768, 3072]
#     cos_sim.append(output)
#
# # print(data[0])
# # print(data[1])
# # print(data[2])
# print(cos_sim[0].shape)
# ax.boxplot(cos_sim, showfliers=False)
# # ax.set_xticklabels(["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"])
# plt.show()





# # ######################################
# # # box plot  (GPT2)
# # ######################################
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
#
# gpt2_weights = np.load('gpt2_weights.npz')
# # for k, v in gpt2_weights.items():
# #     print(k, v.shape)
# fig, ax = plt.subplots()
#
# data = []
# cos_sim = []
# for i in range(12):
#     # query_weights = gpt2_weights[f'encoder.layer.{i}.attention.self.query.weight']
#     # key_weights = gpt2_weights[f'encoder.layer.{i}.attention.self.key.weight']
#     # value_weights = gpt2_weights[f'encoder.layer.{i}.attention.self.value.weight']
#     # proj_weights = gpt2_weights[f'encoder.layer.{i}.attention.output.dense.weight']
#     fc1 = gpt2_weights[f'h.{i}.mlp.c_fc.weight'].T  # [3072, 768]
#     fc2 = gpt2_weights[f'h.{i}.mlp.c_proj.weight']  # [3072, 768]
#     data += [fc2.reshape(-1)]
#
#     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#     # output = cos(torch.FloatTensor(fc1), torch.FloatTensor(fc2))
#     output = cos(torch.FloatTensor(fc1).T, torch.FloatTensor(fc2).T)  # [768, 3072]
#     cos_sim.append(output)
#
# # print(data[0])
# # print(data[1])
# # print(data[2])
# print(cos_sim[0].shape)
# ax.boxplot(cos_sim, showfliers=False)
# # ax.set_xticklabels(["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"])
# plt.show()






