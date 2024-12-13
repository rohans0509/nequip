from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from . import builder_utils


def SimpleIrrepsConfig(config, prefix: Optional[str] = None):
    """Builder that pre-processes options to allow "simple" configuration of irreps."""

    # We allow some simpler parameters to be provided, but if they are,
    # they have to be correct and not overridden
    simple_irreps_keys = ["l_max", "parity", "num_features"]
    real_irreps_keys = [
        "chemical_embedding_irreps_out",
        "feature_irreps_hidden",
        "irreps_edge_sh",
        "conv_to_output_hidden_irreps_out",
    ]

    prefix = "" if prefix is None else f"{prefix}_"

    has_simple: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in simple_irreps_keys
    )
    has_full: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in real_irreps_keys
    )
    assert has_simple or has_full

    update = {}
    if has_simple:
        # nothing to do if not
        lmax = config.get(f"{prefix}l_max", config["l_max"])
        parity = config.get(f"{prefix}parity", config["parity"])
        num_features = config.get(f"{prefix}num_features", config["num_features"])
        update[f"{prefix}chemical_embedding_irreps_out"] = repr(
            o3.Irreps([(num_features, (0, 1))])  # n scalars
        )
        update[f"{prefix}irreps_edge_sh"] = repr(
            o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
        )
        update[f"{prefix}feature_irreps_hidden"] = repr(
            o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        )
        update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
            # num_features // 2  scalars
            o3.Irreps([(max(1, num_features // 2), (0, 1))])
        )

    # check update is consistant with config
    # (this is necessary since it is not possible
    #  to delete keys from config, so instead of
    #  making simple and full styles mutually
    #  exclusive, we just insist that if full
    #  and simple are provided, full must be
    #  consistant with simple)
    for k, v in update.items():
        if k in config:
            assert (
                config[k] == v
            ), f"For key {k}, the full irreps options had value `{config[k]}` inconsistant with the value derived from the simple irreps options `{v}`"
        config[k] = v


def EnergyModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    num_layers = config.get("num_layers", 3)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer

    # .update also maintains insertion order
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear,
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
        }
    )

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )


def LayerwiseIrrepsConfig(config, prefix: Optional[str] = None):
    """Builder that configures different irreps for each layer.
    
    Config parameters:
    - layerwise_irreps: List of dicts, one per layer, each containing:
        - l_max: maximum angular momentum
        - num_features: number of features
        - parity: whether to use parity (optional)
    """
    prefix = "" if prefix is None else f"{prefix}_"
    
    # Get layer configurations
    layer_configs = config.get(f"{prefix}layerwise_irreps", [])
    num_layers = len(layer_configs)
    
    if num_layers == 0:
        raise ValueError("Must specify at least one layer in layerwise_irreps")
        
    update = {}
    
    # Chemical embedding uses first layer's config
    first_layer = layer_configs[0]
    update[f"{prefix}chemical_embedding_irreps_out"] = repr(
        o3.Irreps([(first_layer["num_features"], (0, 1))])
    )
    
    # Configure each layer's irreps
    for i, layer_cfg in enumerate(layer_configs):
        l_max = layer_cfg["l_max"]
        num_features = layer_cfg["num_features"]
        parity = layer_cfg.get("parity", True)
        
        # Build irreps for this layer
        layer_irreps = []
        for l in range(l_max + 1):
            # Add both parities if parity is True
            if parity:
                layer_irreps.extend([
                    (num_features, (l, 1)),   # even parity
                    (num_features, (l, -1))   # odd parity
                ])
            else:
                layer_irreps.append((num_features, (l, 1)))
                
        update[f"{prefix}feature_irreps_hidden_{i}"] = repr(o3.Irreps(layer_irreps))
    
    # Edge attributes use maximum l_max across all layers
    max_l_max = max(cfg["l_max"] for cfg in layer_configs)
    update[f"{prefix}irreps_edge_sh"] = repr(
        o3.Irreps.spherical_harmonics(max_l_max, p=1)
    )
    
    # Output uses final layer's configuration
    final_layer = layer_configs[-1]
    update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
        o3.Irreps([(final_layer["num_features"], (0, 1))])
    )
    
    # Update config
    for k, v in update.items():
        if k in config:
            assert config[k] == v, f"Inconsistent irreps specification for {k}"
        config[k] = v
    
    return config
