import torch


class SolverStat(dict):
    """
    A class for storing solver statistics.

    This class is a subclass of dict, which allows users to query the solver statistics as dictionary keys.

    Valid Keys:
        - ``'abs_lowest'``: 
            The lowest absolute fixed point errors achieved, i.e. :math:`\|z - f(z)\|`.
            torch.Tensor of shape :math:`(B,)`.
        - ``'rel_lowest'``: 
            The lowest relative fixed point errors achieved, i.e., :math:`\|z - f(z)\| / \|f(z)\|`. 
            torch.Tensor of shape :math:`(B,)`.
        - ``'abs_trace'``:
            The absolute fixed point errors achieved along the solver steps.
            torch.Tensor of shape :math:`(B, N)`, where :math:`N` is the solver step consumed.
        - ``'rel_trace'``:
            The relative fixed point errors achieved along the solver steps.
            torch.Tensor of shape :math:`(B, N)`, where :math:`N` is the solver step consumed.
        - ``'nstep'``:
            The number of step when the fixed point errors were achieved.
            torch.Tensor of shape :math:`(B,)`.
        - ``'sradius'``: 
            Optional. The largest (abs.) eigenvalue estimated by power method.
            Available in the eval mode when ``sradius_mode`` set to ``True``.
            torch.Tensor of shape :math:`(B,)`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self['abs_lowest'] = self.get('abs_lowest', torch.tensor([-1.]))
        self['rel_lowest'] = self.get('rel_lowest', torch.tensor([-1.]))
        self['abs_trace'] = self.get('abs_trace', torch.tensor([[-1.]]))
        self['rel_trace'] = self.get('rel_trace', torch.tensor([[-1.]]))
        self['nstep'] = self.get('nstep', torch.tensor([-1.]))
        self['sradius'] = self.get('sradius', torch.tensor([-1.]))

    @classmethod
    def from_solver_info(cls, stop_mode, lowest_dict, trace_dict, lowest_step_dict):
        """
        Generates a SolverStat object from solver information.

        Args:
            stop_mode (str): Mode of stopping criteria ('rel' or 'abs').
            lowest_dict (dict): Dictionary storing the lowest differences.
            trace_dict (dict): Dictionary to trace absolute and relative differences.
            lowest_step_dict (dict): Dictionary storing the steps at which the lowest differences occurred.

        Returns:
            SolverStat: A SolverStat object containing solver statistics.
        """
        info = {
            'abs_lowest': lowest_dict['abs'],
            'rel_lowest': lowest_dict['rel'],
            'abs_trace': torch.stack(trace_dict['abs'], dim=1),
            'rel_trace': torch.stack(trace_dict['rel'], dim=1),
            'nstep': lowest_step_dict[stop_mode], 
        }
        return cls(**info)

    @classmethod
    def from_final_step(cls, z, fz, nstep=0):
        """
        Generates a SolverStat object from final-step solver statistics.

        Args:
            z (torch.Tensor): Final fixed point estimate.
            fz (torch.Tensor): Function evaluation of final fixed point estimate.
            nstep (int, optional): Total number of steps in the solver. Default 0.

        Returns:
            SolverStat: A SolverStat object with final-step solver statistics.
        """
        if not torch.is_tensor(z):
            z = torch.cat([zi.flatten(start_dim=1) for zi in z], dim=1)
            fz = torch.cat([fi.flatten(start_dim=1) for fi in fz], dim=1)

        diff = fz - z
        abs_lowest = diff.flatten(start_dim=1).norm(dim=1)
        rel_lowest = abs_lowest / (fz.flatten(start_dim=1).norm(dim=1) + 1e-8)
        nstep = nstep * torch.ones(z.shape[0], device=z.device)
        info = {
            'abs_lowest': abs_lowest,
            'rel_lowest': rel_lowest,
            'abs_trace': abs_lowest.unsqueeze(dim=1),
            'rel_trace': rel_lowest.unsqueeze(dim=1),
            'nstep': nstep, 
        }
        return cls(**info)