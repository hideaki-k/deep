import torch

def data_sampler(batch_size, num_points):
    half_batch_size = int(batch_size/2)
    normal_sampled = torch.randn(half_batch_size, num_points, 3)
    uniform_sampled = torch.rand(half_batch_size, num_points, 3)
    normal_labels = torch.ones(half_batch_size)
    uniform_labels = torch.zeros(half_batch_size)
    
    input_data = torch.cat((normal_sampled, uniform_sampled), dim=0) #[64, 64, 3]
    labels = torch.cat((normal_labels, uniform_labels), dim=0)
    print('input_data_',input_data.shape)

    data_shuffle = torch.randperm(batch_size)

    return input_data[data_shuffle].view(-1, 3), labels[data_shuffle].view(-1, 1)