from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Action:
    action_id: str
    description: str
    cost: float


def get_available_actions() -> List[Action]:
    """
    Fixed, business-approved action catalog.
    No ML logic here.
    """
    return [
        Action(
            action_id="DISCOUNT_10",
            description="Offer 10% monthly discount",
            cost=50.0
        ),
        Action(
            action_id="PRIORITY_SUPPORT",
            description="Provide priority customer support",
            cost=30.0
        ),
        Action(
            action_id="LOYALTY_OFFER",
            description="Loyalty retention offer",
            cost=20.0
        ),
    ]