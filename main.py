from eval import eval
from train import train
from utils import parse_args


if __name__ == '__main__':
    
    # Paths
    dataset_root = "../egocart/"
    train_path = dataset_root + "train_set/"
    train_rgb_path = train_path + "train_RGB/"
    test_path = dataset_root + "test_set/"
    test_rgb_path = test_path + "test_RGB/"

    paths = {
        "train_path": train_path,
        "train_rgb_path": train_rgb_path,
        "test_path": test_path,
        "test_rgb_path": test_rgb_path
    }

    # Argument parsing  
    args = parse_args()

    if args.mode == "train":
       train(args, paths)
    elif args.mode == "eval":
        eval(args, paths)
    elif args.mode == "explore":
        # load train data in a dataframe 
        # print some statistics and distributions about the dataset
        # visualize some samples
        None
