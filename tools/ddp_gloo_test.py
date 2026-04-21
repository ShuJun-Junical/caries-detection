import torch
import torch.distributed as dist


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    print('rank', rank, 'initialized')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
