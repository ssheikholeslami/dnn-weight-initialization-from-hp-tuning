from random import seed, randint
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master-seed', type=int,
                        help='the master random seed to generate other random seeds)')
    parser.add_argument('-n', '--number-of-seeds', type=int,
                        help='number of random seeds (numbers) to generate')
    args = parser.parse_args()

    seed(args.master_seed)
    seeds = []
    for _ in range(args.number_of_seeds):
        # print(randint(0, 100000))
        seeds.append(randint(0, 100000))
    
    print(f"Master seed: {args.master_seed}")
    print(seeds)
if __name__ == '__main__':
    main()