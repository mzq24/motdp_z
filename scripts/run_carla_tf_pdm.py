"""Wrapper: mock CARLA deps then run PDM scoring with official LEAD agent."""
import sys, types

# Mock CARLA deps BEFORE any other imports
def _mk(n, p=None):
    m = types.ModuleType(n); sys.modules[n] = m
    if p: setattr(sys.modules[p], n.split('.')[-1], m)
    return m

ag = _mk('agents')
an = _mk('agents.navigation', 'agents')
alp = _mk('agents.navigation.local_planner', 'agents.navigation')
class _R: pass
alp.RoadOption = _R
ac = _mk('agents.navigation.controller', 'agents.navigation')
class _V: pass
ac.VehiclePIDController = _V
c = _mk('carla')
class _L:
    pass

class _Rt:
    pass

class _T:
    pass

class _V3:
    pass
c.Location = _L; c.Rotation = _Rt; c.Transform = _T; c.Vector3D = _V3
_mk('lead.common.ransac')

# Now we can run the PDM scoring
from navsim.planning.script.run_pdm_score_one_stage import main as pdm_main

if __name__ == '__main__':
    pdm_main()
