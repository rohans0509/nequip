import logging

from nequip.data import AtomicDataDict
from nequip.nn import (
    GraphModuleMixin,
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ForceOutput,
    PerSpeciesShift,
    ConvNetLayer,
)
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)


def EnergyModel(**shared_params) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    num_layers = shared_params.pop("num_layers", 3)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "feature_embedding": AtomwiseLinear,
    }

    # add convnet layers
    # we get any before and after layers:
    before_layer = shared_params.pop("before_layer", {})
    after_layer = shared_params.pop("after_layer", {})
    if len(set(before_layer.keys()).union(after_layer.keys())) > 1:
        raise ValueError(
            "before_layer and after_layer may not have common module names"
        )
    # insertion preserves order
    for layer_i in range(num_layers):
        layers.update({f"layer{layer_i}_{bk}": v for bk, v in before_layer.items()})
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer
        layers.update({f"layer{layer_i}_{ak}": v for ak, v in after_layer.items()})

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
            "total_energy_sum": (
                AtomwiseReduce,
                dict(
                    reduce="sum",
                    field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    out_field="raw_total_energy",
                ),
            ),
            "energy_shift": (
                PerSpeciesShift,
                dict(
                    field="raw_total_energy",
                    out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
                ),
            ),
        }
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=shared_params,
        layers=layers,
        irreps_in={AtomicDataDict.POSITIONS_KEY: "1o"},
    )


def ForceModel(**shared_params) -> GraphModuleMixin:
    """Base default energy and force model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.

    A convinience method, equivalent to constructing ``EnergyModel`` and passing it to ``nequip.nn.ForceOutput``.
    """
    energy_model = EnergyModel(**shared_params)
    return ForceOutput(energy_model=energy_model)
