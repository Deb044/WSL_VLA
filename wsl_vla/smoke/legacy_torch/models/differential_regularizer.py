"""
models/differential_regularizer.py

Implements Section 1.3 of Methodology:
Differential Regularization During Sequential Adaptation.

Constrains fragile vision and language components more tightly than robust action components:
    gamma_vis, gamma_lang > gamma_act

Formulation:
    L_ref(Z) = L_task(h_psi(Z)) + sum_m gamma_m * || Pi_shell(Z_m) - Z_m^(0) ||_F^2
where:
    Pi_shell(Z) = R * Z / ||Z||_F (Hyperspherical-shell projection)
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def hyperspherical_shell_projection(z: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Pi_shell(Z) = R * Z / ||Z||_F
    Projects latent tensor Z onto the hyperspherical shell of radius R.
    """
    frobenius_norm = torch.norm(z, p="fro", dim=(-2, -1), keepdim=True) + 1e-8
    return radius * (z / frobenius_norm)


class DifferentialRegularizer(nn.Module):
    """
    Applies modality-asymmetric regularization during latent-space refinement:
      - gamma_vis:  Regularization coefficient for vision sub-modules
      - gamma_lang: Regularization coefficient for language sub-modules
      - gamma_act:  Regularization coefficient for action sub-modules
    Satisfies: gamma_vis, gamma_lang > gamma_act
    """
    def __init__(
        self,
        schedule: str = "fixed",
        gamma_vis: float = 1.0,
        gamma_lang: float = 1.0,
        gamma_act: float = 0.2,
        shell_radius: float = 1.0,
    ):
        super().__init__()
        self.schedule = schedule.lower()
        self.gamma_vis = gamma_vis
        self.gamma_lang = gamma_lang
        self.gamma_act = gamma_act
        self.shell_radius = shell_radius

        # Internal tracking for drift-informed and online adaptive schedules
        self.running_drifts = {"vis": 0.0, "lang": 0.0, "act": 0.0}

    def get_gammas(self) -> Dict[str, float]:
        """Returns active gamma regularization weights for each modality."""
        if self.schedule == "fixed":
            return {
                "vis": self.gamma_vis,
                "lang": self.gamma_lang,
                "act": self.gamma_act,
            }
        elif self.schedule == "drift_informed":
            # Proportional to drift sensitivity: boost fragile components if drift is detected
            v_mult = 1.0 + min(self.running_drifts["vis"] * 2.0, 3.0)
            l_mult = 1.0 + min(self.running_drifts["lang"] * 2.0, 3.0)
            a_mult = 1.0 + min(self.running_drifts["act"] * 0.5, 1.5)
            return {
                "vis": self.gamma_vis * v_mult,
                "lang": self.gamma_lang * l_mult,
                "act": self.gamma_act * a_mult,
            }
        elif self.schedule == "adaptive":
            # Live drift re-scaling (EWC-like in latent space)
            total_d = sum(self.running_drifts.values()) + 1e-6
            w_v = (self.running_drifts["vis"] / total_d) * 3.0
            w_l = (self.running_drifts["lang"] / total_d) * 3.0
            w_a = (self.running_drifts["act"] / total_d) * 1.0
            return {
                "vis": max(self.gamma_vis, w_v),
                "lang": max(self.gamma_lang, w_l),
                "act": min(self.gamma_act, max(0.05, w_a)),
            }
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def update_drift(self, z_current: Dict[str, torch.Tensor], z_initial: Dict[str, torch.Tensor]):
        """Updates drift statistics for adaptive schedules."""
        with torch.no_grad():
            for m in ["vis", "lang", "act"]:
                if m in z_current and m in z_initial:
                    drift_val = torch.norm(z_current[m] - z_initial[m], p=2).item()
                    self.running_drifts[m] = 0.8 * self.running_drifts[m] + 0.2 * drift_val

    def compute_regularization_loss(
        self,
        z_current: Dict[str, torch.Tensor],
        z_initial: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_reg = sum_m gamma_m * || Pi_shell(Z_m) - Z_m^(0) ||_F^2
        """
        gammas = self.get_gammas()
        loss_reg = torch.tensor(0.0, device=z_current["vis"].device)
        modal_losses = {}

        for m in ["vis", "lang", "act"]:
            z_m = z_current[m]
            z0_m = z_initial[m]

            # Hyperspherical shell projection
            z_proj = hyperspherical_shell_projection(z_m, radius=self.shell_radius)
            z0_proj = hyperspherical_shell_projection(z0_m, radius=self.shell_radius)

            # Frobenius norm distance
            dist_f = torch.sum((z_proj - z0_proj) ** 2)
            weighted_dist = gammas[m] * dist_f

            loss_reg = loss_reg + weighted_dist
            modal_losses[f"reg_{m}"] = weighted_dist.item()
            modal_losses[f"gamma_{m}"] = gammas[m]

        return loss_reg, modal_losses
