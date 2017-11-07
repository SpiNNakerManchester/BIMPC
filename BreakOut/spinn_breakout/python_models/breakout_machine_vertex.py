from enum import Enum

# PACMAN imports
from pacman.model.decorators.overrides import overrides


# SpinnFrontEndCommon imports
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.interface.provenance \
    .provides_provenance_data_from_machine_impl \
    import ProvidesProvenanceDataFromMachineImpl


# ----------------------------------------------------------------------------
# BreakoutMachineVertex
# ----------------------------------------------------------------------------
class BreakoutMachineVertex(MachineVertex):
    _BREAKOUT_REGIONS = Enum(
        value="_BREAKOUT_REGIONS",
        names=[('SYSTEM', 0),
               ('BREAKOUT', 1),
               ('PROVENANCE', 2)])

    def __init__(self, resources_required, constraints=None, label=None):
        # Superclasses
        MachineVertex.__init__(self, label,
                               constraints=constraints)
        # ProvidesProvenanceDataFromMachineImpl.__init__(
        #     self, self._BREAKOUT_REGIONS.PROVENANCE.value, 0)
        self._resource_required = resources_required

    @property
    def resources_required(self):
        return self._resource_required
