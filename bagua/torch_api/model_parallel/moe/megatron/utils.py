def _add_moe_args(parser):
    group = parser.add_argument_group(title="moe")

    group.add_argument(
        "--num-local-experts", type=int, default=0, help="num of local experts"
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="default=1, top-k gating value, only supports k=1 or k=2.",
    )

    return parser
