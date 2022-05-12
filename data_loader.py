"""
Modified version of PyVertical's dataloader for vertically partitioned data.
"""
from typing import List
from typing import Tuple
from uuid import UUID

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate



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


class VerticalDataLoader:
    """Dataloader which batches data from a complete
    set of vertically-partitioned datasets
    i.e. the images dataset AND the labels dataset
    """

    def __init__(self, dataset, *args, **kwargs):

        # Split datasets
        self.partition_image1, self.partition_image2, self.partition_image3, self.partition_image4, self.partition_labels = partition_dataset(
        #self.data_partition_data, self.data_partition_labels = partition_dataset(
            dataset, remove_data=False, keep_order=False
        )

        assert self.partition_image1.targets is None
        assert self.partition_image2.targets is None
        assert self.partition_image3.targets is None
        assert self.partition_image4.targets is None
        assert self.partition_labels.data is None

        self.dataloader_image1 = SinglePartitionDataLoader( self.partition_image1, *args, **kwargs )
        self.dataloader_image2 = SinglePartitionDataLoader( self.partition_image2, *args, **kwargs )
        self.dataloader_image3 = SinglePartitionDataLoader( self.partition_image3, *args, **kwargs )
        self.dataloader_image4 = SinglePartitionDataLoader( self.partition_image4, *args, **kwargs )
        self.dataloader_labels = SinglePartitionDataLoader( self.partition_labels, *args, **kwargs )

    def __iter__(self):   
        return zip(self.dataloader_image1, self.dataloader_image2, self.dataloader_image3, self.dataloader_image4, self.dataloader_labels)

    def __len__(self):
        return sum(len(x) for x in [self.dataloader_image1, self.dataloader_image2, self.dataloader_image3, self.dataloader_image4, self.dataloader_labels]) // 5

    def drop_non_intersecting(self, intersection: List[int]):
        """Remove elements and ids in the datasets that are not in the intersection."""
        for dataloader in [self.dataloader_image1, self.dataloader_image2, self.dataloader_image3, self.dataloader_image4]:
            dataloader.dataset.data = dataloader.dataset.data[intersection]
            dataloader.dataset.ids  = dataloader.dataset.ids[intersection]

        self.dataloader_labels.dataset.targets = self.dataloader_labels.dataset.targets[intersection]
        self.dataloader_labels.dataset.ids     = self.dataloader_labels.dataset.ids[intersection]

    def sort_by_ids(self) -> None:
        """
        Sort each dataset by ids
        """
        for dataloader in [self.dataloader_image1, self.dataloader_image2, self.dataloader_image3, self.dataloader_image4, self.dataloader_labels]:
            dataloader.dataset.sort_by_ids()



def partition_dataset(
    dataset: Dataset,
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

    partition_image1 = deepcopy(dataset)
    partition_image2 = deepcopy(dataset)
    partition_image3 = deepcopy(dataset)
    partition_image4 = deepcopy(dataset)
    partition_labels = deepcopy(dataset)

    # Partition data
    for partition in [partition_image1, partition_image2, partition_image3, partition_image4]:
        partition.targets = None
        partition.data = partition.data[:,:14,:14]
    partition_labels.data = None

    # Re-index data
    idxs = []
    for partition in [partition_image1, partition_image2, partition_image3, partition_image4, partition_labels]:
        idxs.append( np.arange(len(partition)) )

    # Remove random subsets of data with 1% prob
    if remove_data:
        for i, partition in enumerate([partition_image1, partition_image2, partition_image3, partition_image4, partition_labels]):
            idxs[i] = np.random.uniform(0, 1, len(partition)) > 0.01

    if not keep_order:
        for idx in idxs:
            np.random.shuffle(idx)

    for i, partition in enumerate([partition_image1, partition_image2, partition_image3, partition_image4, partition_labels]):
        partition.data = partition.data[idxs[i]]
        partition.ids  = partition.ids[idxs[i]]

    return partition_image1, partition_image2, partition_image3, partition_image4, partition_labels
