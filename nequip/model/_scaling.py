import logging
from typing import List, Optional

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin, PerSpeciesScaleShift
from nequip.data import AtomicDataDict, AtomicDataset


def RescaleEnergyEtc(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """

    global_scale = config.get(
        "global_rescale_scale",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        if AtomicDataDict.FORCE_KEY in model.irreps_out
        else f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
    )
    # TODO: change this default?
    global_shift = config.get(
        "global_rescale_shift", f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean"
    )

    # = Get statistics of training dataset =
    if initialize:
        str_names = []
        for value in [global_scale, global_shift]:
            if isinstance(value, str):
                str_names += [value]
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, torch.Tensor)
            ):
                # valid values
                pass
            else:
                raise ValueError(f"Invalid global scale `{value}`")

        # = Compute shifts and scales =
        computed_stats = _compute_stats(
            str_names=str_names,
            dataset=dataset,
            stride=config.dataset_statistics_stride,
        )

        if isinstance(global_scale, str):
            global_scale = computed_stats[str_names.index(global_scale)]
        if isinstance(global_shift, str):
            global_shift = computed_stats[str_names.index(global_shift)]

        RESCALE_THRESHOLD = 1e-6
        if isinstance(global_scale, float) and global_scale < RESCALE_THRESHOLD:
            raise ValueError(
                f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
            )

        logging.debug(
            f"Initially outputs are globally scaled by: {global_scale}, total_energy are globally shifted by {global_shift}."
        )
    else:
        # Put dummy values
        if global_shift is not None:
            global_shift = 0.0  # it has some kind of value
        if global_scale is not None:
            global_scale = 1.0  # same,

    # == Build the model ==
    return RescaleOutput(
        model=model,
        scale_keys=[
            k
            for k in (
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.PER_ATOM_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
            )
            if k in model.irreps_out
        ],
        scale_by=global_scale,
        shift_keys=[
            k for k in (AtomicDataDict.TOTAL_ENERGY_KEY,) if k in model.irreps_out
        ],
        shift_by=global_shift,
        trainable_global_rescale_shift=config.get(
            "trainable_global_rescale_shift", False
        ),
        trainable_global_rescale_scale=config.get(
            "trainable_global_rescale_scale", False
        ),
    )


def PerSpeciesRescale(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
):
    """Add global rescaling for energy(-based quantities).

    If ``initialize`` is false, doesn't compute statistics.
    """
    module_prefix = "PerSpeciesScaleShift_"

    force_training = AtomicDataDict.FORCE_KEY in model.irreps_out

    # = Determine energy rescale type =
    # TO DO, how to make the default consistent with the global scale function?
    global_scale = config.get(
        "global_rescale_scale",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        if force_training
        else f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_std",
    )

    # TODO: how to make the default consistent with rescale?
    global_shift = config.get(
        "global_rescale_shift", f"dataset_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean"
    )
    scales = config.get(module_prefix + "scales", None)
    shifts = config.get(module_prefix + "shifts", None)
    trainable = config.get(module_prefix + "trainable", False)
    kwargs = config.get(module_prefix + "kwargs", {})

    if global_shift is not None:
        if trainable or not (scales is None and shifts is None):
            logging.warning(
                f"!!!! Careful global_shift is set to {global_shift}."
                f"This is not a good set up with per species shifts: {shifts}"
                f"and scales: {scales} that are trainable={trainable}"
            )
    if not trainable:
        if scales is None and shifts is None:
            return model
        elif scales == 1.0 and shifts == 0.0:
            return model

    logging.info(f"Enable per species scale/shift")

    # = Determine what statistics need to be compute =
    if initialize:
        str_names = []
        for value in [scales, shifts, global_scale]:
            if isinstance(value, str):
                str_names += [value]
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, list)
                or isinstance(value, torch.Tensor)
            ):
                # valid values
                pass
            else:
                raise ValueError(f"Invalid value `{value}`")

        # = Compute shifts and scales =
        computed_stats = _compute_stats(
            str_names=str_names,
            dataset=dataset,
            stride=config.dataset_statistics_stride,
            kwargs=kwargs,
        )

        if isinstance(scales, str):
            scales = computed_stats[str_names.index(scales)]

        if isinstance(shifts, str):
            shifts = computed_stats[str_names.index(shifts)]

        if isinstance(global_scale, str):
            global_scale = computed_stats[str_names.index(global_scale)]

        if global_scale is not None:
            if scales is not None:
                scales = scales / global_scale
            if shifts is not None:
                shifts = shifts / global_scale

    else:
        # Put dummy values
        scales = None
        shifts = None

    # insert in per species shift
    model.insert_from_parameters(
        before="total_energy_sum",
        name="per_species_scale_shift",
        shared_params=config,
        builder=PerSpeciesScaleShift,
        params=dict(
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            num_types=config.num_types,
            shifts=shifts,
            scales=scales,
            trainable=trainable,
        ),
    )

    logging.debug(f"Atomic outputs are scaled by: {scales}, shifted by {shifts}.")

    # == Build the model ==
    return model


def _compute_stats(
    str_names: List[str], dataset, stride: int, kwargs: Optional[dict] = {}
):
    """return the values of statistics over dataset
    quantity name should be dataset_key_stat, where key can be any key
    that exists in the dataset, stat can be mean, std

    Args:

    str_names: list of strings that define the quantity to compute
    dataset: dataset object to run the stats over
    stride: # frames to skip for every one frame to include
    """

    # parse the list of string to field, mode
    # and record which quantity correspond to which computed_item
    stat_modes = []
    stat_fields = []
    stat_strs = []
    ids = []
    tuple_ids = []
    tuple_id_map = {"mean": 0, "std": 1, "rms": 0}
    input_kwargs = {}
    for name in str_names:

        # remove dataset prefix
        if name.startswith("dataset_"):
            name = name[len("dataset_") :]
        # identify per_species and per_atom modes
        prefix = ""
        if name.startswith("per_species_"):
            name = name[len("per_species_") :]
            prefix = "per_species_"
        elif name.startswith("per_atom_"):
            name = name[len("per_atom_") :]
            prefix = "per_atom_"

        stat = name.split("_")[-1]
        field = "_".join(name.split("_")[:-1])
        if stat in ["mean", "std"]:
            stat_mode = prefix + "mean_std"
            stat_str = field + prefix + "mean_std"
        elif stat in ["rms"]:
            stat_mode = prefix + "rms"
            stat_str = field + prefix + "rms"
        else:
            raise ValueError(f"Cannot handle {stat} type quantity")

        if stat_str in stat_strs:
            ids += [stat_strs.index(stat_str)]
        else:
            ids += [len(stat_strs)]
            stat_strs += [stat_str]
            stat_modes += [stat_mode]
            stat_fields += [field]
            if stat_mode.startswith("per_species_"):
                if field in kwargs:
                    input_kwargs[field + stat_mode] = kwargs[field]
        tuple_ids += [tuple_id_map[stat]]

    values = dataset.statistics(
        fields=stat_fields,
        modes=stat_modes,
        stride=stride,
        kwargs=input_kwargs,
    )
    return [values[idx][tuple_ids[i]] for i, idx in enumerate(ids)]
