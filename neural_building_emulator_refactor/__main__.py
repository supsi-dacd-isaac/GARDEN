"""List models available in the refactored comparison package."""

from .registry import MODEL_REGISTRY


def main() -> None:
    print("Available models:")
    for name, spec in MODEL_REGISTRY.items():
        print(f"  {name:45s} task={spec.task:14s} {spec.description}")


if __name__ == "__main__":
    main()
