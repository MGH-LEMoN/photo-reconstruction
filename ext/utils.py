import random
import subprocess

import numpy
import torch


def seed_all(seed):
    # https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    seed = 0 if not seed else seed
    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_git_revision_branch() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )

    # return (
    #     subprocess.check_output(["git", "branch", "--show-current"])
    #     .decode("ascii")
    #     .strip()
    # )


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_git_revision_url():
    return f"https://github.com/MGH-LEMoN/Photo-SynthSeg/tree/{get_git_revision_short_hash()}"


if __name__ == "__main__":
    print(get_git_revision_short_hash())
