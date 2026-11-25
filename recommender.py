import random
from abc import ABC, abstractmethod


class RecommendationStrategy(ABC):
    """Abstract Base Class (Week 8 OOP) for product recommendation strategies."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def score(self, rule):
        """Child classes must override this method to choose their ranking metric."""
        pass

    def recommend(self, basket, rules_df, top_n=3):
        """Matches rules to a basket, excludes existing items, and returns top N."""
        if rules_df is None or rules_df.empty:
            return []

        basket_set = {item.strip().title() for item in basket if item.strip()}
        candidates = {}

        for _, rule in rules_df.iterrows():
            # Check if all 'IF' items (antecedents) are inside the user's basket
            if rule["antecedents"].issubset(basket_set):
                rule_score = self.score(rule)

                for item in rule["consequents"]:
                    # Recommend only items NOT already in the basket
                    if item not in basket_set:
                        if item not in candidates or rule_score > candidates[item]["score"]:
                            candidates[item] = {
                                "product": item,
                                "score": round(float(rule_score), 3),
                                "confidence": round(float(rule["confidence"]), 3),
                                "lift": round(float(rule["lift"]), 3),
                                "based_on": ", ".join(sorted(rule["antecedents"])),
                            }

        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]


class ConfidenceStrategy(RecommendationStrategy):
    """Ranks recommendations by highest Confidence."""

    def __init__(self):
        super().__init__("Confidence Strategy")

    def score(self, rule):
        return rule["confidence"]


class LiftStrategy(RecommendationStrategy):
    """Ranks recommendations by highest Lift."""

    def __init__(self):
        super().__init__("Lift Strategy")

    def score(self, rule):
        return rule["lift"]


class StrategyEvaluator:
    """Compares strategies by hiding 1 item per test basket and measuring Hit Rate@3."""

    def __init__(self, pipeline, test_ratio=0.25, seed=42):
        self._pipeline = pipeline
        self._test_ratio = test_ratio
        self._seed = seed

    def evaluate(self, baskets_dict, strategies, top_n=3):
        valid_baskets = [items for items in baskets_dict.values() if len(items) >= 2]
        rng = random.Random(self._seed)
        rng.shuffle(valid_baskets)

        split = int(len(valid_baskets) * (1 - self._test_ratio))
        train_baskets = valid_baskets[:split]
        test_baskets = valid_baskets[split:]

        # Mine rules ONLY from training baskets so test baskets stay unseen
        train_rules = self._pipeline.mine_rules(train_baskets)

        # Hide 1 random item from each test basket
        test_cases = []
        for basket in test_baskets:
            hidden = rng.choice(sorted(list(basket)))
            test_cases.append((basket - {hidden}, hidden))

        results = {}
        for strategy in strategies:
            hits = 0
            for partial_basket, hidden_item in test_cases:
                recs = strategy.recommend(partial_basket, train_rules, top_n=top_n)
                if hidden_item in [r["product"] for r in recs]:
                    hits += 1

            total = len(test_cases)
            hit_rate = round((hits / total) * 100, 1) if total > 0 else 0.0
            results[strategy.name] = {"hits": hits, "total": total, "hit_rate": hit_rate}

        return results