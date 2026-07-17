"""Product recommendation, the third model.

The real one trains on the merged dataset and is blocked until that lands. This is the interface
plus a stub, so the CLI can be built and tested now.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    product: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.product} ({self.confidence:.0%} confidence)"


class Recommender:
    """Base class. Subclasses predict the product a customer is most likely to buy."""

    is_stub: bool = False

    def recommend(self, customer_id: str) -> Recommendation:
        raise NotImplementedError


class StubRecommender(Recommender):
    """Placeholder until the trained model exists. Announces itself so it cannot reach the demo."""

    is_stub = True

    PRODUCTS: list[str] = ["Wireless Earbuds", "Running Shoes", "Coffee Maker", "Backpack"]

    def recommend(self, customer_id: str) -> Recommendation:
        index = sum(ord(c) for c in str(customer_id)) % len(self.PRODUCTS)
        return Recommendation(product=self.PRODUCTS[index], confidence=0.0)
