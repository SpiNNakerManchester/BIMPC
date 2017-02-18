from spynnaker.pyNN.spinnaker import executable_finder
from model_binaries import __file__ as binaries_path
from python_models.breakout import Breakout
from visualiser.visualiser import Visualiser
from visualiser.visualiser_subsamp import Visualiser_subsamp
import os

# This adds the model binaries path to the paths searched by sPyNNaker
executable_finder.add_path(os.path.dirname(binaries_path))
