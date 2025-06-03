from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class DataLoader(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch.

    This class handles loading and processing of data for a given dataset,
    initializing with specific constraints and reconstruction settings.

    """

    def __init__(
        self,
        id: Union[np.ndarray, List[int]],
        initrecon: bool,
        example: str,
        initial_regul: Optional[str] = None,
        test_train: Optional[str] = None,
    ) -> None:
        """
        Initializes the DataLoader with the given parameters.

        Args:
            id (List[int]): List of sample indices.
            initrecon (bool): Flag to determine whether to use initial
            reconstruction or FBP data.
            test_train (str): Indicates whether the data is for 'test',
            'train', or 'validation'.
            example (str): Which example (e.g. lodopab or synthetic).
        """
        self.id = id
        self.initrecon = initrecon
        self.test_train = test_train
        self.example = example
        self.initial_regul = initial_regul

        if example == "synthetic":
            self.X = np.load("data/data_synthetic/phantom.npy")
            if self.initrecon:
                self.Y = np.load(
                    f"data/data_synthetic/" f"init_regul_{initial_regul}.npy"
                )
            else:
                if self.initial_regul == "fbp":
                    self.Y = np.load("data/data_synthetic/fbp.npy")
                elif self.initial_regul == "landweber":
                    self.Y = np.load("data/data_synthetic/landweber.npy")
                else:
                    raise ValueError("Initial reconstruction not found.")
            self.Z = np.load("data/data_synthetic/limited_angle_data.npy")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.id)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates one sample of data.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the
            input data (X),
            target data (Y), and limited angle data (Z), each with an added
            channel dimension.
        """
        idx = self.id[index]

        if self.example == "lodopab":
            # Load the data from individual files
            p = "data/data_lodopab/data_processed/"
            base_path = f"{p}{self.test_train}/single_files/"
            X = np.load(f"{base_path}phantom_{idx}.npy")
            Z = np.load(f"{base_path}limited_angle_data_{idx}.npy")

            if self.initrecon:
                Y = np.load(f"{base_path}init_regul_{self.initial_regul}_{idx}.npy")
            else:
                if self.initial_regul == "fbp":
                    Y = np.load(f"{base_path}fbp_{idx}.npy")
                elif self.initial_regul == "landweber":
                    Y = np.load(f"{base_path}landweber_{idx}.npy")
                else:
                    raise ValueError("Initial reconstruction not found.")

        elif self.example == "synthetic":
            X = self.X[index, :, :]
            Y = self.Y[index, :, :]
            Z = self.Z[index, :, :]

        # Add channel dimension to the arrays
        X = X[None, :, :]
        Y = Y[None, :, :]
        Z = Z[None, :, :]

        return X, Y, Z
