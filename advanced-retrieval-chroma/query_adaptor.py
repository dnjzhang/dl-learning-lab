import numpy as np
import torch
from datasets import tqdm

#adapter_query_embeddings = torch.Tensor(np.array(adapter_query_embeddings))
#adapter_doc_embeddings = torch.Tensor(np.array(adapter_doc_embeddings))
#adapter_labels = torch.Tensor(np.expand_dims(np.array(adapter_labels),1))
# Fake data
adapter_query_embeddings = torch.tensor([[1, 2, 3], [4, 5, 6]])
adapter_doc_embeddings =  torch.tensor([[1, 2, 3], [4, 5, 6]])
adapter_labels=  torch.tensor([[1, 2, 3], [4, 5, 6]])

dataset = torch.utils.data.TensorDataset(adapter_query_embeddings, adapter_doc_embeddings, adapter_labels)


def model(query_embedding, document_embedding, adaptor_matrix):
    updated_query_embedding = torch.matmul(adaptor_matrix, query_embedding)
    return torch.cosine_similarity(updated_query_embedding, document_embedding, dim=0)


def mse_loss(query_embedding, document_embedding, adaptor_matrix, label):
    return torch.nn.MSELoss()(model(query_embedding, document_embedding, adaptor_matrix), label)


# Initialize the adaptor matrix
mat_size = len(adapter_query_embeddings[0])
adapter_matrix = torch.randn(mat_size, mat_size, requires_grad=True)

min_loss = float('inf')
best_matrix = None

for epoch in tqdm(range(100)):
    for query_embedding, document_embedding, label in dataset:
        loss = mse_loss(query_embedding, document_embedding, adapter_matrix, label)

        if loss < min_loss:
            min_loss = loss
            best_matrix = adapter_matrix.clone().detach().numpy()

        loss.backward()
        with torch.no_grad():
            adapter_matrix -= 0.01 * adapter_matrix.grad
            adapter_matrix.grad.zero_()

