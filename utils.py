import os, sys



def load_from_dotenv(variable_name):
    ''' The dotenv module is incompatible with some PyVertical dependencies. This function can be used to load environment variables instead. It throws an exception if 
    '''
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    if path[-1] != '/': path += '/'
    dotenv_file = path + '.env'
    with open(dotenv_file,'r') as f:
        for line in f:
            if variable_name in line:
                try: 
                    try:
                        return line.split('"')[1]
                    except:
                        return line.split("'")[1]
                except IndexError:
                    raise EnvironmentVariableNotSetError(f'The environment variable {variable_name} is not available in the file {dotenv_file}')


"""
Handling vertically partitioned data
"""
from copy import deepcopy
from typing import List
from typing import Tuple
from typing import TypeVar
from uuid import uuid4

from PIL import Image
import numpy as np

# Ignore errors when running mypy script
# mypy: ignore-errors

Dataset = TypeVar("Dataset")


def add_ids(cls):
    """Decorator to add unique IDs to a dataset

    Args:
        cls (torch.utils.data.Dataset) : dataset to generate IDs for

    Returns:
        VerticalDataset : A class which wraps cls to add unique IDs as an attribute,
            and returns data, target, id when __getitem__ is called
    """

    class VerticalDataset(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.ids = np.array([uuid4() for _ in range(len(self))])

        def __getitem__(self, index):
            if self.data is None:
                img = None
            else:
                img = self.data[index]
                img = Image.fromarray(img.numpy(), mode="L")

                if self.transform is not None:
                    img = self.transform(img)

            if self.targets is None:
                target = None
            else:
                target = int(self.targets[index]) if self.targets is not None else None

                if self.target_transform is not None:
                    target = self.target_transform(target)

            id = self.ids[index]

            # Return a tuple of non-None elements
            return (*filter(lambda x: x is not None, (img, target, id)),)

        def __len__(self):
            if self.data is not None:
                return self.data.size(0)
            else:
                return len(self.targets)

        def get_ids(self) -> List[str]:
            """Return a list of the ids of this dataset."""
            return [str(id_) for id_ in self.ids]

        def sort_by_ids(self):
            """
            Sort the dataset by IDs in ascending order
            """
            ids = self.get_ids()
            sorted_idxs = np.argsort(ids)

            if self.data is not None:
                self.data = self.data[sorted_idxs]

            if self.targets is not None:
                self.targets = self.targets[sorted_idxs]

            self.ids = self.ids[sorted_idxs]

    return VerticalDataset
