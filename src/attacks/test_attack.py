from synonym_attack import SynonymAttack
import time

TEXT = """The researchers developed an method to improve 
the performance of the computer system. The approach provides
significant benefits for complex computational problems"""

def test_attack(method, probability=0.5, seed=42):
    print("\n" + "=" * 60)
    print(f"Testing: {method.upper()} SYNONYM ATTACK")
    print("=" * 60)

    attack = SynonymAttack(
        method=method,
        p=probability,
        seed=seed
    )

    start = time.time()

    try:
        result = attack.warp(TEXT)

        print("\nOriginal text:")
        print(TEXT)

        print("\nAttacked text:")
        print(result)

        print(f"\nExecution time: {time.time() - start:.2f}s")

        # Reproducibility check
        result2 = attack.warp(TEXT, p=probability, seed=seed)

        print("\nReproducibility:")
        print("PASS" if result == result2 else "FAIL")

        print("\nText changed:", result != TEXT)

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":

    # Test WordNet
    test_attack("wordnet", probability=1)

    # Test Qwen
    test_attack("llm", probability=1)

    print("\nAll tests completed.")