"""Product recommendation — the third model.

The real implementation trains on Anthony's merged dataset and is blocked until that lands. What
exists now is the interface plus a stub, so the CLI can be built and tested end to end today and
the trained model can drop in without the app changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Recommendation:
    product: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.product} ({self.confidence:.0%} confidence)"


@runtime_checkable
class Recommender(Protocol):
    def recommend(self, customer_id: str) -> Recommendation:
        """Predict the product this customer is most likely to buy."""
        ...


class StubRecommender:
    """Placeholder standing in for the trained model.

    Deterministic per customer so the CLI behaves consistently while it is being developed, and
    loud about being fake so a stub can never quietly end up in the demo.
    """

    PRODUCTS = ["Wireless Earbuds", "Running Shoes", "Coffee Maker", "Backpack"]

    def __init__(self) -> None:
        self.is_stub = True

    def recommend(self, customer_id: str) -> Recommendation:
        idx = sum(ord(c) for c in str(customer_id)) % len(self.PRODUCTS)
        return Recommendation(product=self.PRODUCTS[idx], confidence=0.0)
