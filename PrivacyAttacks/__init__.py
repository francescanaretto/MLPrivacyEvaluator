"""
This module includes the implemented privacy attacks.
"""

from ._mia_privacy_attack import MiaPrivacyAttack
from ._aloa_privacy_attack import AloaPrivacyAttack


__all__ = [
    "MiaPrivacyAttack",
    "AloaPrivacyAttack"
]
