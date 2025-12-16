import math
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Callable, Any, Dict, List, Union
from torch import Tensor
from torch import device as TorchDevice
from pathlib import Path

# We assume this base class exists in your environment as per your request
from app.services.base_explainer import BaseExplainer

# ==========================================
# PART 1: UTILS (src/shats/utils.py)
# ==========================================

class StrategySubsets(Enum):
    """Enum representing the subsets generation strategy."""
    EXACT = 1
    APPROX = 2

Subset = tuple[int, ...]
Predictors2subsetsDict = dict[tuple[int, int], tuple[list[Subset], list[Subset]]]

class GeneratedSubsets(NamedTuple):
    predictors_to_subsets: Predictors2subsetsDict
    all_subsets: list[Subset]

def _calculate_num_subsets_to_generate(
    total_groups: int, total_wanted_subsets: int, size: int, strategy: StrategySubsets
):
    num = math.floor(
        total_wanted_subsets
        * (size + 1) ** (2 / 3)
        / sum([(k + 1) ** (2 / 3) for k in range(total_groups)])
    )
    num = min(num, math.comb(total_groups - 1, size))

    if strategy.value == StrategySubsets.EXACT.value:
        num = math.comb(total_groups - 1, size)

    if num == 0:
        num = 1

    return num

def generate_subsets(
    total_groups: int,
    total_wanted_subsets: int,
    strategy: StrategySubsets = StrategySubsets.APPROX,
) -> GeneratedSubsets:
    if total_groups < 1:
        raise ValueError("nGroups must be at least 1.")
    if total_wanted_subsets < 0:
        raise ValueError("nSubsets must be non-negative.")

    all_subsets: list[set[Subset]] = [set() for _ in range(total_groups + 1)]
    subset_dict = {}

    for group in range(total_groups):
        for size in range(total_groups):
            num_of_subsets_to_generate = _calculate_num_subsets_to_generate(
                total_groups, total_wanted_subsets, size, strategy
            )

            # Generate subsets
            subsets_without_group = [
                subset for subset in all_subsets[size] if group not in subset
            ]
            subsets_with_group: list[tuple[int, ...]] = [
                tuple(sorted(subset + (group,))) for subset in subsets_without_group
            ]

            remaining_nums = list(range(total_groups))
            remaining_nums.remove(group)

            # Avoid duplicates by maintaining intersections
            intersection = list[Subset]()

            for i, subset in enumerate(subsets_without_group):
                if subsets_with_group[i] in all_subsets[size + 1]:
                    intersection.append(subset)

            subsets_without_group = sorted(
                subsets_without_group, key=lambda x: x in intersection, reverse=False
            )
            subsets_with_group = sorted(
                subsets_with_group, key=lambda x: x in intersection, reverse=False
            )

            while len(subsets_without_group) < num_of_subsets_to_generate:
                random_subset_without = tuple(
                    sorted(random.sample(remaining_nums, size))
                )
                random_subset_with = tuple(sorted(random_subset_without + (group,)))

                if random_subset_without not in all_subsets[size]:
                    all_subsets[size].add(random_subset_without)
                    subsets_without_group.append(random_subset_without)
                    subsets_with_group.append(random_subset_with)

                if random_subset_with not in all_subsets[size + 1]:
                    all_subsets[size + 1].add(random_subset_with)

            subsets_with_group = subsets_with_group[:num_of_subsets_to_generate]

            for subset in subsets_with_group:
                if subset not in all_subsets[size + 1]:
                    all_subsets[size + 1].add(subset)

            subsets_without_group = subsets_without_group[:num_of_subsets_to_generate]
            subset_dict[(group, size)] = (
                [list(subset) for subset in subsets_with_group],
                [list(subset) for subset in subsets_without_group],
            )

    # Flatten all subsets
    flatenned_subsets = [
        tuple(subset) for sizeSubsets in all_subsets for subset in sizeSubsets
    ]

    return GeneratedSubsets(subset_dict, flatenned_subsets)

def estimate_m(total_features: int, total_desired_subsets: int) -> int:
    limit = (
        2
        * sum((i + 1) ** (2 / 3) for i in range(total_features))
        / total_features ** (2 / 3)
    )
    limit = round(limit)

    if total_desired_subsets <= limit:
        return limit

    step = max((limit**2 - limit) // 20, 1)
    values = range(limit, limit**2, step)
    list_values = list(values)

    sizes = list[int]()

    for value in list_values:
        _, subsets_total = generate_subsets(total_features, value)
        sizes.append(len(subsets_total))

    x = np.array(list_values)
    y = np.array(sizes)

    # Calculate regression coefficients
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate slope (m) and intersection (b)
    numer = np.sum((x - mean_x) * (y - mean_y))
    denom = np.sum((x - mean_x) ** 2)
    slope = numer / denom
    intercept = mean_y - slope * mean_x

    # Calculate final `m`
    m = (total_desired_subsets - intercept) / slope

    if np.isinf(m) or np.isnan(m):
        return limit

    if m < 0:
        return limit

    return round(m)


# ==========================================
# PART 2: GROUPING (src/shats/grouping.py)
# ==========================================

class GroupingPlotTexts(NamedTuple):
    """
    Named tuple to store the plot texts for grouping strategies.
    """
    title: str
    y_label: str
    columns: list[str]


class AbstractGroupingStrategy(ABC):
    """
    Abstract class for grouping strategies in ShaTS.
    """
    _default_names_key: str
    _plot_title: str
    _plot_y_label: str

    def __init__(self, groups_num: int | None = None, names: list[str] | None = None) -> None:
        if groups_num is None and names is None:
            raise ValueError("groups_num or names must be provided")
        if groups_num is not None and names is not None:
            if groups_num != len(names):
                raise ValueError(
                    "If groups_num and names are provided, they must match in length"
                )
        if groups_num is None and names is not None:
            groups_num = len(names)
        self.groups_num = groups_num
        self._names = names

    @abstractmethod
    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        raise NotImplementedError()

    def get_plot_texts(self) -> GroupingPlotTexts:
        names = self._names or [
            f"{self._default_names_key}{i+1}" for i in range(self.groups_num)
        ]
        return GroupingPlotTexts(self._plot_title, self._plot_y_label, names)


class TimeGroupingStrategy(AbstractGroupingStrategy):
    """
    Grouping strategy based on time.
    """
    _default_names_key = "instant"
    _plot_title = "ShaTS (Temporal)"
    _plot_y_label = "Time"

    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        indexes = torch.tensor(list(subset), dtype=torch.long, device=device)
        tensor[:, indexes, :] = support_tensor[:, indexes, :].clone()
        return tensor.clone()


class FeaturesGroupingStrategy(AbstractGroupingStrategy):
    """
    Grouping strategy based on features.
    """
    _default_names_key = "feature"
    _plot_title = "ShaTS(Feature)"
    _plot_y_label = "Feature"

    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        indexes = torch.tensor(list(subset), dtype=torch.long, device=device)
        for instant in range(support_tensor[0].shape[0]):
            tensor[:, instant, indexes] = support_tensor[:, instant, indexes].clone()
        return tensor.clone()


class MultifeaturesGroupingStrategy(AbstractGroupingStrategy):
    """
    Grouping strategy based on multiple features.
    """
    _default_names_key = "multifeature"
    _plot_title = "ShaTS (MULTIFEATURE)"
    _plot_y_label = "MULTIFEATURE"

    def __init__(
        self,
        custom_groups: list[list[int]],
        groups_num: int | None = None,
        names: list[str] | None = None,
    ) -> None:
        super().__init__(groups_num, names)
        self._custom_groups = custom_groups

    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        all_indexes = list[int]()

        for group in subset:
            all_indexes.extend(self._custom_groups[group])
        indexes_tensor = torch.tensor(all_indexes, dtype=torch.long, device=device)

        tensor[:, :, indexes_tensor] = support_tensor[:, :, all_indexes].clone()

        return tensor


# ==========================================
# PART 3: SHATS CORE (src/shats/shats.py)
# ==========================================

class ShaTS(ABC):
    """
    Abstract class for initializing ShaTS module
    """
    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: Union[str, AbstractGroupingStrategy],
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: Union[str, torch.device, int] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):

        self.support_dataset = support_dataset
        self.support_tensor = torch.stack(
            [data for data in support_dataset]
        ).to(device)
        self.window_size = support_dataset[0].shape[0]
        self.num_of_features = support_dataset[0].shape[1]
        self.subsets_generation_strategy = subsets_generation_strategy
        self.grouping_strategy : AbstractGroupingStrategy

        if isinstance(grouping_strategy, AbstractGroupingStrategy):
            self.grouping_strategy = grouping_strategy

        elif grouping_strategy == "time":
            self.grouping_strategy = TimeGroupingStrategy(
                groups_num=self.window_size
            )
        elif grouping_strategy == "feature":
            self.grouping_strategy = FeaturesGroupingStrategy(
                groups_num=self.num_of_features
            )
        elif grouping_strategy == "multifeature":
            if custom_groups is None:
                raise ValueError(
                    "custom_groups must be provided when grouping_strategy is 'multifeature'."
                )
            self.grouping_strategy = MultifeaturesGroupingStrategy(
                groups_num=len(custom_groups),
                custom_groups=custom_groups,
            )
        else:
            raise ValueError(
                "grouping_strategy must be 'time', 'feature', or 'multifeature'."
            )

        self.device = device
        self.batch_size = batch_size
        self.nclass = model_wrapper(support_dataset[0].unsqueeze(0).to(device)).shape[1]
        self.model_wrapper = model_wrapper

        self.m = estimate_m(self.groups_num, m)

        self.subsets_dict, self.all_subsets = generate_subsets(
            self.groups_num, self.m, self.subsets_generation_strategy
        )

        keys_support_subsets = [
            (tuple(subset), entity)
            for subset in self.all_subsets
            for entity in range(len(self.support_dataset))
        ]
        self.pair_dicts = {
            (subset, entity): i
            for i, (subset, entity) in enumerate(keys_support_subsets)
        }

        self.coefficients_dict = self._generate_coefficients_dict()
        self.mean_prediction = self._compute_mean_prediction()

    @property
    def groups_num(self) -> int:
        """Returns the number of groups."""
        return self.grouping_strategy.groups_num

    @abstractmethod
    def compute(
        self,
        test_dataset: list[Tensor]
    ) -> Tensor:
        raise NotImplementedError()

    def plot(
        self,
        shats_values: Tensor,
        test_dataset: list[Tensor] | None = None,
        predictions: Tensor | None = None,
        path: str | Path | None = None,
        segment_size: int = 100,
        class_to_explain: int = 0,
    ):
        if test_dataset is None and predictions is None:
            raise ValueError(
                "Either test_dataset or predictions must be provided."
            )
        if test_dataset is not None and predictions is not None:
            raise ValueError(
                "Only one of test_dataset or predictions should be provided."
            )
        elif predictions is not None:
            model_predictions = predictions
        elif test_dataset is not None:
            model_predictions = torch.zeros(
                len(test_dataset), device=self.device
            )
            for i, data in enumerate(test_dataset):
                model_predictions[i] = self.model_wrapper(data.unsqueeze(0).to(self.device))[0][class_to_explain]
        
        shats_values = shats_values[:,:, class_to_explain]
        fontsize = 25
        size = shats_values.shape[0]

        arr_plot = np.zeros((self.groups_num, size))
        arr_prob = np.zeros(size)

        for i in range(size):
            arr_plot[:, i] = shats_values[i].cpu().numpy()
            arr_prob[i] = model_predictions[i]

        vmin, vmax = -0.5, 0.5
        cmap = plt.get_cmap("bwr")

        n_segments = (size + segment_size - 1) // segment_size
        fig, axs = plt.subplots(
            n_segments, 1, figsize=(15, 25 * (max(10, self.groups_num) / 36) * n_segments)
        ) 

        if n_segments == 1:
            axs = [axs]

        for n in range(n_segments):
            real_end = min((n + 1) * segment_size, size)
            if n == n_segments - 1:
                real_end = arr_plot.shape[1]
                arr_plot = np.hstack(
                    (
                        arr_plot,
                        np.zeros(
                            (self.groups_num, segment_size - (size % segment_size))
                        ),
                    )
                )
                arr_prob = np.hstack(
                    (arr_prob, -np.ones(segment_size - (size % segment_size)))
                )
                size = arr_plot.shape[1]

            init = n * segment_size
            end = min((n + 1) * segment_size, size)
            segment = arr_plot[:, init:end]
            ax = axs[n]

            ax.set_xlabel("Window", fontsize=fontsize)

            cax = ax.imshow(
                segment,
                cmap=cmap,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            cbar_ax = fig.add_axes(
                (
                    ax.get_position().x1 + 0.15,
                    ax.get_position().y0 - 0.05,
                    0.05,
                    ax.get_position().height + 0.125,
                )
            )

            cbar = fig.colorbar(cax, cax=cbar_ax, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontsize)

            ax2 = ax.twinx()

            prediction = arr_prob[init:real_end] 
            ax2.plot(
                np.arange(0, real_end - init),
                prediction,
                linestyle="--",
                color="darkviolet",
                linewidth=4,
            )

            ax2.axhline(0.5, color="black", linewidth=1, linestyle="--")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", labelsize=fontsize)

            ax2.set_ylabel("Model outcome", fontsize=fontsize)

            legend = ax2.legend(
                ["Model outcome", "Threshold"],
                fontsize=fontsize,
                loc="lower left",
                bbox_to_anchor=(0.0, -0.0),
            )
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((0, 0, 0, 0))
            legend.get_frame().set_edgecolor("black")

            title, y_label, columns_labels = self.grouping_strategy.get_plot_texts()

            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)

            ax.set_yticks(np.arange(self.groups_num))
            ax.set_yticklabels(columns_labels, fontsize=fontsize)

            xticks = np.arange(0, segment.shape[1], 5)
            xlabels = np.arange(init, real_end, 5)

            xticks = xticks[: len(xlabels)]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=fontsize)

        if path is not None:
            plt.savefig(path)
        plt.show()

    def _generate_coefficients_dict(self) -> dict[int, float]:
        coef_dict = dict[int, float]()
        if self.subsets_generation_strategy.value == StrategySubsets.EXACT.value:
            for i in range(self.groups_num):
                coef_dict[i] = (
                    math.factorial(i)
                    * math.factorial(self.groups_num - i - 1)
                    / math.factorial(self.groups_num)
                )
        else:
            for i in range(self.groups_num):
                coef_dict[i] = 1 / self.groups_num
        return coef_dict

    def _compute_mean_prediction(self) -> Tensor:
        mean_prediction = torch.zeros(self.nclass, device=self.device)

        with torch.no_grad():
            for data in self.support_dataset:
                # Add batch dim
                probs = self.model_wrapper(data.unsqueeze(0).to(self.device))
                for class_idx in range(self.nclass):
                    mean_prediction[class_idx] += probs[0, class_idx].cpu()

        return mean_prediction / len(self.support_dataset)

    def _modify_data_batches(self, data: Tensor) -> list[Tensor]:
        modified_data_batches = list[Tensor]()

        for subset in self.all_subsets:
            data_tensor = (
                data
                .unsqueeze(0)
                .expand(len(self.support_dataset), *data.shape)
                .clone()
                .to(self.device)
            )
            modified_data_batches.append(
                self.grouping_strategy.modify_tensor(
                    subset, self.device, self.support_tensor, data_tensor
                )
            )

        return modified_data_batches

    def _compute_probs(
        self,
        modified_data_batches: list[Tensor]
    ) -> list[Tensor]:
        probs: list[list[Tensor]] = []
        probs = [[] for _ in range(self.nclass)]

        for i in range(0, len(modified_data_batches), self.batch_size):
            batch = torch.cat(modified_data_batches[i : i + self.batch_size]).to(self.device)
            batch_probs = self.model_wrapper(batch)

            for class_idx in range(self.nclass):
                class_probs = batch_probs[:, class_idx].cpu()
                probs[class_idx].append(class_probs)

        probs = [torch.cat(class_probs, dim=0).to(self.device) for class_probs in probs]

        return probs

    def _compute_differences(
        self, probs: Tensor, instant: int, size: int
    ) -> tuple[Tensor, Tensor]:
        subsets_with, subsets_without = self.subsets_dict[(instant, size)]
        prob_with = torch.zeros(self.nclass, len(subsets_with), device=self.device)
        prob_without = torch.zeros(self.nclass, len(subsets_without), device=self.device)

        for i, (item_with, item_without) in enumerate(
            zip(subsets_with, subsets_without)
        ):
            indexes_with = [
                self.pair_dicts[(tuple(item_with), entity)]
                for entity in range(len(self.support_dataset))
            ]
            indexes_without = [
                self.pair_dicts[(tuple(item_without), entity)]
                for entity in range(len(self.support_dataset))
            ]
            indexes_with_tensor = torch.tensor(indexes_with,
                                             dtype=torch.long, device=self.device)
            indexes_without_tensor = torch.tensor(indexes_without,
                                                dtype=torch.long, device=self.device)
            coef = self.coefficients_dict[len(item_without)]
            mean_probs_with = torch.zeros(self.nclass, device=self.device)
            mean_probs_without = torch.zeros(self.nclass, device=self.device)

            for class_idx in range(self.nclass):
                selected_probs_with = torch.index_select(probs[class_idx],
                                                       0, indexes_with_tensor)
                selected_probs_without = torch.index_select(probs[class_idx],
                                                          0, indexes_without_tensor)

                mean_probs_with[class_idx] = selected_probs_with.mean() * coef
                mean_probs_without[class_idx] = selected_probs_without.mean() * coef

            prob_with[:, i] = mean_probs_with
            prob_without[:, i] = mean_probs_without

        return prob_with, prob_without


class ApproShaTS(ShaTS):
    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: Union[str, AbstractGroupingStrategy],
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: Union[str, torch.device, int] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):
        super().__init__(
            support_dataset=support_dataset,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=subsets_generation_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
            model_wrapper=model_wrapper
        )

    def compute(
        self,
        test_dataset: list[Tensor]
    ) -> Tensor:
        shats_values_list = torch.zeros(
        len(test_dataset),
        self.groups_num,
        self.nclass,
        device=self.device
        )
        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                # progress = (idx + 1) / total * 100
                # print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                tsgshapvalues = torch.zeros(self.groups_num, self.nclass, device=self.device)

                modified_data_batches = self._modify_data_batches(data)
                probs = self._compute_probs(modified_data_batches)

                for group in range(self.groups_num):
                    for size in range(self.groups_num):
                        prob_with, prob_without = self._compute_differences(
                            probs, group, size
                        )

                        resta = prob_with - prob_without

                        for class_idx in range(self.nclass):
                            tsgshapvalues[group, class_idx] += resta[class_idx].mean()

                shats_values_list[idx] = tsgshapvalues.clone()

                del (
                    modified_data_batches,
                    probs,
                    tsgshapvalues,
                )
                torch.cuda.empty_cache()

        return shats_values_list


class FastShaTS(ShaTS):
    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: Union[str, AbstractGroupingStrategy],
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: Union[str, torch.device, int] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):
        super().__init__(
            support_dataset=support_dataset,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=subsets_generation_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
            model_wrapper=model_wrapper
        )

    def compute(
        self,
        test_dataset: list[Tensor]
    ) -> Tensor:
        tsgshapvalues_list = torch.zeros(
        len(test_dataset),
        self.groups_num,
        self.nclass,
        device=self.device
        )
        reversed_dict = self._reverse_dict(self.subsets_dict, self.all_subsets)

        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                # progress = (idx + 1) / total * 100
                # print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                tsgshapvalues = torch.zeros(self.groups_num, self.nclass, device=self.device)

                modified_data_batches = self._modify_data_batches(data)

                probs = self._compute_probs(modified_data_batches)

                for class_idx in range(self.nclass):

                    for i, value in enumerate(reversed_dict.values()):
                        add = probs[class_idx][
                            i
                            * len(self.support_dataset) : (i + 1)
                            * len(self.support_dataset)
                        ].mean()

                        add = add / self.groups_num
                        for v in value:

                            if v[1] == 0:
                                tsgshapvalues[v[0][0]][class_idx] -= add / len(
                                    self.subsets_dict[(v[0][0], v[0][1])][0]
                                )

                            else:
                                tsgshapvalues[v[0][0]][class_idx] += add / len(
                                    self.subsets_dict[(v[0][0], v[0][1])][0]
                                )
                    tsgshapvalues_list[idx] = -tsgshapvalues.clone()

                del (
                    modified_data_batches,
                    probs,
                    tsgshapvalues,
                )

        return tsgshapvalues_list

    def _reverse_dict(
        self,
        subsets_dict: dict[Any, Any],
        subsets_total: list[Any]
    ) -> dict[tuple[Any], list[Any]]:
        subsets_dict_reversed = dict[tuple[Any], list[Any]]()
        for subset in subsets_total:
            subsets_dict_reversed[tuple(subset)] = []

        for subset in subsets_total:
            for clave, valor in subsets_dict.items():
                if list(subset) in valor[0]:
                    subsets_dict_reversed[tuple(subset)].append((clave, 0))

                if list(subset) in valor[1]:
                    subsets_dict_reversed[tuple(subset)].append((clave, 1))

        return subsets_dict_reversed


class KernelShaTS(ShaTS):
    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: Union[str, AbstractGroupingStrategy],
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: Union[str, torch.device, int] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):
        super().__init__(
            support_dataset=support_dataset,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=subsets_generation_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
            model_wrapper=model_wrapper
        )

        self.keys_support_subsets = [
            (tuple(subset), entity)
            for subset in self.all_subsets
            for entity in range(len(self.support_dataset))
        ]
        self.pair_dicts = {
            (subset, entity): i
            for i, (subset, entity) in enumerate(self.keys_support_subsets)
        }

        self.binary_vectors = self._subsets_to_binary_vectors()
        self.weights = self._compute_weigths()
 
 
    def compute(
        self,
        test_dataset: list[Tensor],
        learning_rate: float = 0.01,
        early_stopping_rate: float = 0.1,
        num_epochs: int = 10,
    ) -> Tensor:
        # tsgshapvalues_list declaration was moved up in the original logic implicitly
        # but here we must init it.
        # Original code had a bug or redundancy where it initialized it twice.
        
        def weighted_loss(y_true, y_pred, weights):
            return torch.sum(weights * (y_true - y_pred) ** 2)
        
        class WeightedLinearRegression(nn.Module):
            def __init__(self, input_dim):
                super(WeightedLinearRegression, self).__init__()
                self.linear = nn.Linear(input_dim, 1)
        
            def forward(self, x):
                return self.linear(x)
        
        tsgshapvalues_list = torch.zeros(len(test_dataset), self.groups_num, device=self.device)
        regression_model = WeightedLinearRegression(input_dim=self.binary_vectors.shape[1]).to(self.device)
        optimizer = optim.Adam(regression_model.parameters(), learning_rate)
 
        weights_tensor = torch.tensor(self.weights, dtype=torch.float32, device=self.device, requires_grad=False)
 
        last_loss = 0.01
        for idx, data in enumerate(test_dataset):
            with torch.no_grad():
                modified_data_batches = self._modify_data_batches(data)
                probs = self._compute_probs(modified_data_batches)
 
            prediccion = torch.zeros(len(self.all_subsets), device=self.device)
            for i, subset in enumerate(self.all_subsets):
                indexes = [self.pair_dicts[(tuple(subset), entity)] for entity in range(len(self.support_dataset))]
                # Using class 0 prob for regression target? 
                # Original code: probs[0][indexes].mean(). This implies single class explanation or class 0.
                # Assuming class 0 for KernelShaTS based on original snippet provided.
                prediccion[i] = probs[0][indexes].mean()
 
            target = prediccion.unsqueeze(1).to(self.device)
 
            for _ in range(num_epochs):
                optimizer.zero_grad()
                pred = regression_model(torch.tensor(self.binary_vectors, dtype=torch.float32).to(self.device))
 
                loss = weighted_loss(pred, target, weights_tensor)
 
                loss.backward()
                upgrade = abs(last_loss - loss.item()) / last_loss
                if upgrade < early_stopping_rate:
                    break
                last_loss = loss.item()
                optimizer.step()
 
            raw_weights = regression_model.linear.weight.detach().cpu().numpy().flatten()        
 
            tsgshapvalues = raw_weights
            tsgshapvalues_list[idx] = torch.tensor(tsgshapvalues, device=self.device)
 
            del modified_data_batches, probs
            torch.cuda.empty_cache()
 
        return tsgshapvalues_list
 
    def _subsets_to_binary_vectors(self):
        
        binary_vectors = np.zeros((len(self.all_subsets), self.groups_num), dtype=int)
 
        for i, subset in enumerate(self.all_subsets):
            binary_vectors[i, subset] = 1
 
        return binary_vectors
    
    def _compute_weigths(self):
        weights = []
        M = self.groups_num
        
        for coalition in self.binary_vectors:
            z_prime_size = sum(coalition)
            
            if z_prime_size == 0 or z_prime_size == M:
                weights.append(0)
            else:
                weight = (M - 1) / (math.comb(M, z_prime_size) * z_prime_size * (M - z_prime_size))
                weights.append(weight)
        
        return weights


class CachedKernelShaTS:
    """
    Meta-explainer wrapping KernelShaTS with caching.
    """

    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: Union[str, AbstractGroupingStrategy],
        m: int = 5,
        batch_size: int = 32,
        device: Union[str, torch.device, int] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
        global_similarity_threshold: float = 0.01,
        group_change_threshold: float = 0.01,
        max_cache_size: int = 1000,
        hash_decimals: int = 3,
    ) -> None:
        self.base_explainer = KernelShaTS(
            model_wrapper=model_wrapper,
            support_dataset=support_dataset,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=StrategySubsets.APPROX,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
        )

        self.device = device
        self.groups_num = self.base_explainer.groups_num
        self.nclass = self.base_explainer.nclass
        self.grouping_strategy = self.base_explainer.grouping_strategy
        self.global_similarity_threshold = global_similarity_threshold
        self.group_change_threshold = group_change_threshold
        self.max_cache_size = max_cache_size
        self.hash_decimals = hash_decimals

        self._cache: dict[int, Tensor] = {}
        self._cache_keys: list[int] = []  

    def _hash_tensor(self, data: Tensor) -> int:
        arr = data.detach().cpu().numpy()
        arr = np.round(arr, decimals=self.hash_decimals)
        return hash(arr.tobytes())

    def _global_change(self, x_new: Tensor, x_old: Tensor) -> float:
        diff = (x_new - x_old).detach().cpu().numpy()
        old = x_old.detach().cpu().numpy()
        num = np.linalg.norm(diff)
        den = np.linalg.norm(old) + 1e-8
        return float(num / den)

    def _per_group_change(self, x_new: Tensor, x_old: Tensor) -> np.ndarray:
        x_new_np = x_new.detach().cpu().numpy()
        x_old_np = x_old.detach().cpu().numpy()
        changes = np.zeros(self.groups_num, dtype=float)

        if isinstance(self.grouping_strategy, TimeGroupingStrategy):
            for g in range(self.groups_num):
                diff = x_new_np[g, :] - x_old_np[g, :]
                base = x_old_np[g, :]
                num = np.linalg.norm(diff)
                den = np.linalg.norm(base) + 1e-8
                changes[g] = num / den

        elif isinstance(self.grouping_strategy, FeaturesGroupingStrategy):
            for g in range(self.groups_num):
                diff = x_new_np[:, g] - x_old_np[:, g]
                base = x_old_np[:, g]
                num = np.linalg.norm(diff)
                den = np.linalg.norm(base) + 1e-8
                changes[g] = num / den

        elif isinstance(self.grouping_strategy, MultifeaturesGroupingStrategy):
            custom_groups = self.grouping_strategy._custom_groups
            for g, feat_idxs in enumerate(custom_groups):
                diff = x_new_np[:, feat_idxs] - x_old_np[:, feat_idxs]
                base = x_old_np[:, feat_idxs]
                num = np.linalg.norm(diff)
                den = np.linalg.norm(base) + 1e-8
                changes[g] = num / den

        else:
            glob = self._global_change(x_new, x_old)
            changes[:] = glob

        return changes

    def _get_from_cache(self, key: int) -> Tensor | None:
        return self._cache.get(key, None)

    def _store_in_cache(self, key: int, shap: Tensor) -> None:
        if key in self._cache:
            return
        if len(self._cache_keys) >= self.max_cache_size:
            old_key = self._cache_keys.pop(0)
            self._cache.pop(old_key, None)
        self._cache[key] = shap.detach().clone()
        self._cache_keys.append(key)

    def compute(
        self,
        test_dataset: list[Tensor],
        learning_rate: float = 0.01,
        early_stopping_rate: float = 0.1,
        num_epochs: int = 10,
    ) -> Tensor:
        N = len(test_dataset)
        shats_values_list = torch.zeros(
            N, self.groups_num, self.nclass, device=self.device
        )

        prev_x: Tensor | None = None
        prev_shap: Tensor | None = None

        for idx, data in enumerate(test_dataset):
            data = data.to(self.device)
            key = self._hash_tensor(data)

            shap = self._get_from_cache(key)
            if shap is not None:
                shats_values_list[idx] = shap
                prev_x, prev_shap = data, shap
                continue

            if prev_x is not None and prev_shap is not None:
                g_change = self._global_change(data, prev_x)
                if g_change < self.global_similarity_threshold:
                    shats_values_list[idx] = prev_shap
                    self._store_in_cache(key, prev_shap)
                    prev_x, prev_shap = data, prev_shap
                    continue

            base_result = self.base_explainer.compute(
                [data],
                learning_rate=learning_rate,
                early_stopping_rate=early_stopping_rate,
                num_epochs=num_epochs,
            )
            base_shap = base_result[0]

            if base_shap.dim() == 1:
                tmp = torch.zeros(self.groups_num, self.nclass, device=self.device)
                tmp[:, 0] = base_shap
                base_shap = tmp
            elif base_shap.dim() == 2 and base_shap.shape[1] != self.nclass:
                tmp = torch.zeros(self.groups_num, self.nclass, device=self.device)
                cols = min(self.nclass, base_shap.shape[1])
                tmp[:, :cols] = base_shap[:, :cols]
                base_shap = tmp

            if prev_x is not None and prev_shap is not None:
                per_group_change = self._per_group_change(data, prev_x)
                base_np = base_shap.detach().cpu().numpy()
                prev_np = prev_shap.detach().cpu().numpy()

                mask_copy = per_group_change < self.group_change_threshold
                base_np[mask_copy, :] = prev_np[mask_copy, :]

                base_shap = torch.tensor(
                    base_np, device=self.device, dtype=base_shap.dtype
                )

            shats_values_list[idx] = base_shap
            self._store_in_cache(key, base_shap)
            prev_x, prev_shap = data, base_shap

        return shats_values_list


# ==========================================
# PART 4: WRAPPER (ShatsExplainer)
# ==========================================

class ShatsExplainer(BaseExplainer):
    """
    Wrapper class to expose ShaTS functionality via the BaseExplainer interface.
    """
    def explain(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explains the model's predictions using the ShaTS algorithm.

        Args:
            request (Dict[str, Any]): A dictionary containing:
                - "model": Callable, the model to be explained.
                - "data": List or np.array, the input instance(s) to explain.
                - "background_data": List or np.array, the support dataset.
                - "params": (Optional) Dict containing ShaTS configuration:
                    - "grouping_strategy": "time", "feature", or "multifeature" (default "time")
                    - "implementation": "fast", "approx", "kernel", "cached" (default "fast")
                    - "m": int, parameter for subset generation (default 5)
                    - "batch_size": int (default 32)
                    - "custom_groups": List[List[int]] (required if strategy is 'multifeature')

        Returns:
            Dict[str, Any]: A dictionary containing the result:
                - "result": List[List[List[float]]], the ShaTS values.
        """
        
        # 1. Parse Inputs
        model = request.get("model")
        input_data = request.get("data")
        background_data = request.get("background_data")
        params = request.get("params", {})
        
        if model is None or input_data is None or background_data is None:
            raise ValueError("Request must contain 'model', 'data', and 'background_data'.")

        # 2. Configuration
        grouping_strategy = params.get("grouping_strategy", "time")
        implementation = params.get("implementation", "fast")
        m = params.get("m", 5)
        batch_size = params.get("batch_size", 32)
        custom_groups = params.get("custom_groups", None)
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Data Conversion
        # Helper to convert numpy/lists to list[Tensor]
        def to_tensor_list(data_obj) -> list[Tensor]:
            if isinstance(data_obj, np.ndarray):
                # Assume (N, Time, Feat) or (Time, Feat)
                if data_obj.ndim == 2:
                    return [torch.tensor(data_obj, dtype=torch.float32)]
                elif data_obj.ndim == 3:
                    return [torch.tensor(x, dtype=torch.float32) for x in data_obj]
            elif isinstance(data_obj, list):
                # If list of numpy arrays or list of lists
                if len(data_obj) > 0 and isinstance(data_obj[0], (np.ndarray, list)):
                    return [torch.tensor(np.array(x), dtype=torch.float32) for x in data_obj]
            
            raise ValueError("Unsupported data format. Please provide numpy arrays or lists of arrays.")

        test_dataset_tensors = to_tensor_list(input_data)
        support_dataset_tensors = to_tensor_list(background_data)

        # 4. Model Wrapper
        # ShaTS expects a callable that takes a Tensor and returns a Tensor of shape (Batch, Classes)
        # We wrap the user provided model to ensure this.
        def model_wrapper(input_tensor: Tensor) -> Tensor:
            # Move to CPU numpy for generic models, or keep tensor if model is torch
            if isinstance(model, nn.Module):
                model.eval()
                model.to(device)
                with torch.no_grad():
                    return model(input_tensor.to(device))
            else:
                # Generic python function / sklearn model
                arr = input_tensor.detach().cpu().numpy()
                res = model(arr)
                # If model returns pure probabilities/logits
                if isinstance(res, np.ndarray):
                    return torch.tensor(res, dtype=torch.float32, device=device)
                return torch.tensor(np.array(res), dtype=torch.float32, device=device)

        # 5. Select Implementation
        explainer_class = FastShaTS # Default
        if implementation == "approx":
            explainer_class = ApproShaTS
        elif implementation == "kernel":
            explainer_class = KernelShaTS
        elif implementation == "cached":
            explainer_class = CachedKernelShaTS
        
        # 6. Initialize Explainer
        explainer = explainer_class(
            model_wrapper=model_wrapper,
            support_dataset=support_dataset_tensors,
            grouping_strategy=grouping_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups
        )

        # 7. Compute
        # CachedKernelShaTS takes extra params in compute() if provided, but we stick to generic interface here
        result_tensor = explainer.compute(test_dataset_tensors)

        # 8. Format Output
        # Result is (N, Groups, Classes)
        result_list = result_tensor.detach().cpu().numpy().tolist()

        return {"result": result_list}