import matplotlib.pyplot as plt
from pipeline import BasketPipeline
from recommender import ConfidenceStrategy, LiftStrategy, StrategyEvaluator


def save_comparison_chart(results, filename="strategy_comparison.png"):
    """Plots a simple bar chart comparing Hit Rate@3 between strategies (Week 10)."""
    names = list(results.keys())
    rates = [results[n]["hit_rate"] for n in names]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, rates, color=["#4C72B0", "#55A868"], width=0.45)
    plt.title("Recommendation Accuracy (Hit Rate@3)")
    plt.ylabel("Hit Rate (%)")
    plt.ylim(0, 100)

    for bar, val in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[INFO] Chart saved to '{filename}'.")


def main():
    pipeline = BasketPipeline(filepath="transactions.csv", min_support=0.10, min_confidence=0.40, min_lift=1.0)
    strategies = [ConfidenceStrategy(), LiftStrategy()]

    baskets = None
    products = []
    rules_df = None

    while True:
        print("\n=== SHOPPING BASKET ANALYZER ===")
        print("1. Load & Clean Data + Mine Apriori Rules")
        print("2. Get Top-3 Recommendations for a Basket")
        print("3. Evaluate Strategies & Save Chart")
        print("4. Exit")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            baskets, products = pipeline.load_and_clean()
            rules_df = pipeline.mine_rules(list(baskets.values()))

            print(f"\nClean Baskets: {len(baskets)} | Unique Products: {len(products)}")
            print(f"Products: {', '.join(products)}")
            print(f"\nTop Rules (out of {len(rules_df)} found):")

            for idx, (_, row) in enumerate(rules_df.sort_values(by="lift", ascending=False).head(8).iterrows(), 1):
                ant = ", ".join(sorted(row["antecedents"]))
                con = ", ".join(sorted(row["consequents"]))
                print(f"  {idx}. {{{ant}}} -> {{{con}}} | Support: {row['support']:.2f}, Conf: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")

        elif choice == "2":
            if rules_df is None:
                print("[!] Run Option 1 first.")
                continue

            print(f"\nAvailable products: {', '.join(products)}")
            raw = input("Enter basket items separated by commas (e.g., Bread, Butter): ")
            user_basket = {item.strip().title() for item in raw.split(",") if item.strip()}
            print(f"Your Basket: {sorted(list(user_basket))}")

            # Polymorphism: loop over both strategy objects and call the same .recommend() method
            for strategy in strategies:
                recs = strategy.recommend(user_basket, rules_df, top_n=3)
                print(f"\n[{strategy.name}] Top Recommendations:")
                if not recs:
                    print("  No matching rules found.")
                for i, r in enumerate(recs, 1):
                    print(f"  {i}. {r['product']} (Conf: {r['confidence']}, Lift: {r['lift']} | From: {r['based_on']})")

        elif choice == "3":
            if baskets is None:
                print("[!] Run Option 1 first.")
                continue

            evaluator = StrategyEvaluator(pipeline, test_ratio=0.25, seed=42)
            results = evaluator.evaluate(baskets, strategies, top_n=3)

            print("\n--- Strategy Evaluation Results ---")
            for name, m in results.items():
                print(f"{name}: {m['hit_rate']}% Hit Rate ({m['hits']}/{m['total']} test baskets)")

            save_comparison_chart(results)

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("[!] Invalid choice.")


if __name__ == "__main__":
    main()