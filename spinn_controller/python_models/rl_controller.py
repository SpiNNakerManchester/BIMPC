# PACMAN imports
from spynnaker.pyNN.models.common.population_settable_change_requires_mapping import \
    PopulationSettableChangeRequiresMapping

from pacman.executor.injection_decorator import inject_items
from pacman.model.constraints.key_allocator_constraints import ContiguousKeyRangeContraint
from pacman.model.decorators.overrides import overrides
from pacman.model.graphs.application import ApplicationVertex
from pacman.model.resources.cpu_cycles_per_tick_resource import \
    CPUCyclesPerTickResource
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource

# SpinnFrontEndCommon imports
from spinn_front_end_common.abstract_models \
    .abstract_binary_uses_simulation_run import AbstractBinaryUsesSimulationRun
from spinn_front_end_common.abstract_models \
    .abstract_generates_data_specification \
    import AbstractGeneratesDataSpecification
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models. \
    abstract_provides_outgoing_partition_constraints import \
    AbstractProvidesOutgoingPartitionConstraints

from spinn_front_end_common.interface.simulation import simulation_utilities
from spinn_front_end_common.utilities import constants as \
    front_end_common_constants
from spinn_front_end_common.utilities.utility_objs.executable_start_type \
    import ExecutableStartType

# sPyNNaker imports
from spynnaker.pyNN.models.abstract_models import AbstractAcceptsIncomingSynapses
from spynnaker.pyNN.models.neuron import AbstractPopulationVertex
from spynnaker.pyNN.utilities import constants

# Breakout imports
# from breakout_machine_vertex import BreakoutMachineVertex
# Breakout imports
from rl_controller_machine_vertex import RLControllerMachineVertex

# ----------------------------------------------------------------------------
# Breakout
# ----------------------------------------------------------------------------
# **HACK** for Projection to connect a synapse type is required
class BreakoutSynapseType(object):
    def get_synapse_id_by_target(self, target):
        return 0

# ----------------------------------------------------------------------------
# Breakout
# ----------------------------------------------------------------------------
class RLController(
    ApplicationVertex, AbstractGeneratesDataSpecification,
    AbstractHasAssociatedBinary, AbstractProvidesOutgoingPartitionConstraints,
    AbstractAcceptsIncomingSynapses,
    PopulationSettableChangeRequiresMapping):

    def get_connections_from_machine(self, transceiver, placement, edge, graph_mapper, routing_infos,
                                     synapse_information, machine_time_step):
        super(RLController, self).get_connections_from_machine(transceiver, placement, edge, graph_mapper, routing_infos,
                                                           synapse_information, machine_time_step)

    def set_synapse_dynamics(self, synapse_dynamics):
        pass

    def add_pre_run_connection_holder(self, connection_holder, projection_edge, synapse_information):
        super(RLController, self).add_pre_run_connection_holder(connection_holder, projection_edge, synapse_information)

    def get_binary_start_type(self):
        super(RLController, self).get_binary_start_type()
    #
    # def requires_mapping(self):
    #     pass

    def clear_connection_cache(self):
        pass

    BREAKOUT_REGION_BYTES = 4
    WIDTH_PIXELS = 160
    HEIGHT_PIXELS = 128
    COLOUR_BITS = 2

    # **HACK** for Projection to connect a synapse type is required
    synapse_type = BreakoutSynapseType()

    def __init__(self, n_neurons, constraints=None, label="RLController"):
        # **NOTE** n_neurons currently ignored - width and height will be
        # specified as additional parameters, forcing their product to be
        # duplicated in n_neurons seems pointless

        # Superclasses
        ApplicationVertex.__init__(
            self, label, constraints, self.n_atoms)
        AbstractProvidesOutgoingPartitionConstraints.__init__(self)
        PopulationSettableChangeRequiresMapping.__init__(self)

    def get_maximum_delay_supported_in_ms(self, machine_time_step):
        # Breakout has no synapses so can simulate only one time step of delay
        return machine_time_step / 1000.0

    # ------------------------------------------------------------------------
    # ApplicationVertex overrides
    # ------------------------------------------------------------------------
    @overrides(ApplicationVertex.get_resources_used_by_atoms)
    def get_resources_used_by_atoms(self, vertex_slice):
        # **HACK** only way to force no partitioning is to zero dtcm and cpu
        container = ResourceContainer(
            sdram=SDRAMResource(
                self.BREAKOUT_REGION_BYTES +
                front_end_common_constants.SYSTEM_BYTES_REQUIREMENT),
            dtcm=DTCMResource(0),
            cpu_cycles=CPUCyclesPerTickResource(0))

        return container


    @overrides(ApplicationVertex.create_machine_vertex)
    def create_machine_vertex(self, vertex_slice, resources_required,
                              label=None, constraints=None):
        # Return suitable machine vertex
        return RLControllerMachineVertex(resources_required, constraints, label)

    @property
    @overrides(ApplicationVertex.n_atoms)
    def n_atoms(self):
        return 16

    # ------------------------------------------------------------------------
    # AbstractGeneratesDataSpecification overrides
    # ------------------------------------------------------------------------
    @inject_items({"machine_time_step": "MachineTimeStep",
                   "time_scale_factor": "TimeScaleFactor",
                   "graph_mapper": "MemoryGraphMapper",
                   "routing_info": "MemoryRoutingInfos",
                   "tags": "MemoryTags",
                   "n_machine_time_steps": "TotalMachineTimeSteps"})
    @overrides(AbstractGeneratesDataSpecification.generate_data_specification,
               additional_arguments={"machine_time_step", "time_scale_factor",
                                     "graph_mapper", "routing_info", "tags",
                                     "n_machine_time_steps"}
    )
    def generate_data_specification(self, spec, placement, machine_time_step,
                                    time_scale_factor, graph_mapper,
                                    routing_info, tags, n_machine_time_steps):
        vertex = placement.vertex
        vertex_slice = graph_mapper.get_slice(vertex)

        spec.comment("\n*** Spec for RL Controller Instance ***\n\n")
        spec.comment("\nReserving memory space for data regions:\n\n")

        # Reserve memory:
        spec.reserve_memory_region(
            region=RLControllerMachineVertex._BREAKOUT_REGIONS.SYSTEM.value,
                    size=front_end_common_constants.SYSTEM_BYTES_REQUIREMENT,
                    label='setup')
        spec.reserve_memory_region(
            region=RLControllerMachineVertex._BREAKOUT_REGIONS.BREAKOUT.value,
                    size=self.BREAKOUT_REGION_BYTES, label='RLControllerParams')
        # vertex.reserve_provenance_data_region(spec)

        # Write setup region
        spec.comment("\nWriting setup region:\n")
        spec.switch_write_focus(
            RLControllerMachineVertex._BREAKOUT_REGIONS.SYSTEM.value)
        spec.write_array(simulation_utilities.get_simulation_header_array(
            self.get_binary_file_name(), machine_time_step,
            time_scale_factor))

        # Write breakout region containing routing key to transmit with
        spec.comment("\nWriting RL Controller region:\n")
        spec.switch_write_focus(
            RLControllerMachineVertex._BREAKOUT_REGIONS.BREAKOUT.value)
        spec.write_value(routing_info.get_first_key_from_pre_vertex(
            vertex, constants.SPIKE_PARTITION_ID))

        # End-of-Spec:
        spec.end_specification()

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractHasAssociatedBinary.get_binary_file_name)
    def get_binary_file_name(self):
        return "rl_ctrl.aplx"

    @overrides(AbstractHasAssociatedBinary.get_binary_start_type)
    def get_binary_start_type(self):
        return ExecutableStartType.USES_SIMULATION_INTERFACE

    # ------------------------------------------------------------------------
    # AbstractProvidesOutgoingPartitionConstraints overrides
    # ------------------------------------------------------------------------
    @overrides(AbstractProvidesOutgoingPartitionConstraints.
               get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [ContiguousKeyRangeContraint()]
