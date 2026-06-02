from __future__ import annotations

from models.resnet18 import (
    assert_conv_weights_are_ternary,
    build_ternary_resnet18,
    collect_conv_ternary_stats,
)


def main() -> None:
    model = build_ternary_resnet18(
        num_classes=1000,
        seed=0,
        threshold_factor=0.7,
    )

    assert_conv_weights_are_ternary(model)

    stats = collect_conv_ternary_stats(model)

    total_weights = 0
    total_zero = 0

    print("Ternary Conv2d weight statistics")
    print("=" * 100)
    print(
        f"{'layer':35s} {'shape':24s} "
        f"{'-1':>10s} {'0':>10s} {'+1':>10s} {'sparsity':>10s}"
    )
    print("-" * 100)

    for s in stats:
        total_weights += s.total
        total_zero += s.num_zero

        print(
            f"{s.name:35s} "
            f"{str(s.shape):24s} "
            f"{s.num_neg:10d} "
            f"{s.num_zero:10d} "
            f"{s.num_pos:10d} "
            f"{100.0 * s.sparsity:9.2f}%"
        )

    print("-" * 100)
    print(f"Total conv weights: {total_weights:,}")
    print(f"Total zero weights: {total_zero:,}")
    print(f"Overall sparsity:   {100.0 * total_zero / total_weights:.2f}%")
    print("All Conv2d weights are ternary.")


if __name__ == "__main__":
    main()