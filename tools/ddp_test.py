import torch
import torch.distributed as dist


def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    x = torch.tensor([rank], device='cuda')
    dist.all_reduce(x)
    print('rank', rank, 'x', x.item())
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
