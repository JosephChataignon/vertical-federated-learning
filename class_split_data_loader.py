"""
Modified version of PyVertical's dataloader for vertically partitioned data.
"""
from copy import deepcopy
from typing import List
from typing import Tuple
from typing import TypeVar
from uuid import UUID

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

Dataset = TypeVar("Dataset")


def id_collate_fn(batch: Tuple) -> List:
    """Collate data, targets and IDs  into batches

    This custom function is necessary as default collate
    functions cannot handle UUID objects.

    Args:
        batch (tuple of (data, target, id) tuples) : tuple of data returns from each index call
            to the dataset in a batch. To be turned into batched data

    Returns:
        list : List of batched data objects:
            data (torch.Tensor), targets (torch.Tensor), IDs (tuple of strings)
    """
    results = []

    for samples in zip(*batch):
        if isinstance(samples[0], UUID):
            # Turn into a tuple of strings
            samples = (*map(str, samples),)

        # Batch data
        results.append(default_collate(samples))
    return results


class SinglePartitionDataLoader(DataLoader):
    """DataLoader for a single vertically-partitioned dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = id_collate_fn


class ClassSplitDataLoader:
    """Dataloader which batches data from a complete
    set of vertically-partitioned datasets
    i.e. the images dataset AND the labels dataset
    """

    def __init__(self, dataset, class_to_keep, remove_data=False, keep_order=False, *args, **kwargs):

        # Split datasets
        self.data_partition1 = partition_dataset(
            dataset, class_to_keep=class_to_keep, remove_data=False, keep_order=False
        )

        assert self.data_partition1.targets is None

        self.dataloader1 = SinglePartitionDataLoader(
            self.data_partition1, *args, **kwargs
        )

    def __iter__(self):
        return self.dataloader1

    def __len__(self):
        return len(self.dataloader1)

    def drop_non_intersecting(self, intersection: List[int]):
        """Remove elements and ids in the datasets that are not in the intersection."""
        self.dataloader1.dataset.data = self.dataloader1.dataset.data[intersection]
        self.dataloader1.dataset.ids = self.dataloader1.dataset.ids[intersection]


    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        self.dataloader1.dataset.sort_by_ids()



def partition_dataset(
    dataset: Dataset,
    class_to_keep,
    keep_order: bool = False,
    remove_data: bool = True,
):
    """Vertically partition a torch dataset in two

    A vertical partition is when parameters for a single data point is
    split across multiple data holders.
    This function assumes the dataset to split contains images (e.g. MNIST).
    One dataset gets the images, the other gets the labels
    
    THIS IS A MODIFIED VERSION of the PyVertical function partition_dataset,
    here the images are sliced in different parts so that various data holders 
    own different parts of the image.

    When MNIST is loaded by torchvision train/test data and labels are simply named data and targets respectively.

    Args:
        dataset (torch.utils.data.Dataset) : The dataset to split. Must be a dataset of images, containing ids
        keep_order (bool, default = False) : If False, shuffle the elements of each dataset
        remove_data (bool, default = True) : If True, remove datapoints with probability 0.01

    Returns:
        torch.utils.data.Dataset : Dataset containing the first partition: the data/images
        torch.utils.data.Dataset : Dataset containing the second partition: the labels

    Raises:
        RuntimeError : If dataset does not have an 'ids' attribute
        AssertionError : If the size of the provided dataset
            does not have three elements (i.e. is not an image dataset)
    """
    if not hasattr(dataset, "ids"):
        raise RuntimeError("Dataset does not have attribute 'ids'")

    partition1 = deepcopy(dataset)
    
    # Remove data points not belonging to the desired class
    if class_to_keep in np.arange(10):
        print('loading class: '+str(class_to_keep))
        idxs1 = (partition1.targets == class_to_keep)
        partition1.data = partition1.data[idxs1]
        partition1.ids = partition1.ids[idxs1]
    else:
        raise UnexpectedClassException(f'class_to_keep has value {class_to_keep}, which is not one of the classes of this dataset')
    
    
    # Re-index data
    idxs1 = np.arange(len(partition1))

    # Remove random subsets of data with 1% prob
    if remove_data:
        idxs1 = np.random.uniform(0, 1, len(partition1)) > 0.01

    if not keep_order:
        np.random.shuffle(idxs1)

    partition1.data = partition1.data[idxs1]
    partition1.ids = partition1.ids[idxs1]

    # remove labels
    partition1.targets = None
    
    return partition1
