"""Render the active uncertainty pathways in the structured HP emulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class Node:
    key: str
    x: int
    y: int
    width: int
    height: int
    title: str
    lines: tuple[str, ...]
    kind: str


NODES = (
    Node(
        "xi",
        32,
        34,
        190,
        86,
        "Persistent latent xi",
        ("xi(k) ~ N(0, I)", "sampled once; fixed for full trajectory"),
        "persistent",
    ),
    Node(
        "hp_noise",
        390,
        34,
        250,
        86,
        "Local HP randomness",
        ("u_mode(k,t) ~ Uniform(0,1)", "eps_P(k,t) ~ N(0,1)"),
        "local",
    ),
    Node(
        "process_noise",
        740,
        34,
        225,
        86,
        "Thermal process noise",
        ("eps_x(k,t) ~ N(0,I)", "heteroscedastic learned scale"),
        "local",
    ),
    Node(
        "inputs",
        32,
        185,
        196,
        142,
        "Observed conditioning",
        (
            "metadata c",
            "Tset, Tout, solar, ventilation",
            "calendar, availability, force-off",
            "recursive Tin and controller history",
        ),
        "input",
    ),
    Node(
        "initial",
        270,
        165,
        205,
        112,
        "Initial-state generators",
        ("x0 = NN(c, Tin0, xi)", "E0 = positive NN(c, Tin0, xi)", "bounded latent state and energy"),
        "learned",
    ),
    Node(
        "controller",
        270,
        337,
        215,
        150,
        "HP controller + demand",
        (
            "asymmetric Markov p_start, p_stop",
            "pi, log-mu-P, log-sigma-P",
            "monotone requested Qroom",
            "COP(Tout) and buffer loss",
        ),
        "learned",
    ),
    Node(
        "emission",
        535,
        337,
        190,
        150,
        "HP emission",
        (
            "train: E[Pel] = pi E[Pactive]",
            "scenario: m ~ Bernoulli(pi)",
            "Pactive ~ bounded log-normal",
            "Pel = m Pactive",
        ),
        "stochastic",
    ),
    Node(
        "buffer",
        775,
        337,
        190,
        150,
        "Physical buffer",
        (
            "heat = COP Pel",
            "leaky stored energy E",
            "Qroom = min(Qrequest, available)",
            "force-off gates generation only",
        ),
        "physics",
    ),
    Node(
        "outputs",
        1010,
        337,
        160,
        150,
        "Predictions",
        ("Pel_SH", "Qroom", "Tin(t+1)", "particle-specific trajectory"),
        "output",
    ),
    Node(
        "thermal",
        410,
        590,
        270,
        150,
        "Contracting thermal transition",
        (
            "Mt, bt = NN(c, xi, exogenous_t)",
            "x(t+1) = bounded(Mt x(t) + bt + sigma eps_x)",
            "Tfree = decoder(x(t+1), xi)",
            "||Mt|| <= gamma",
        ),
        "learned",
    ),
    Node(
        "q_to_t",
        735,
        590,
        225,
        150,
        "Positive leaky Q-to-T modes",
        (
            "gain + mode weights = NN(c, xi)",
            "fixed tau = 0.25, 1, 4, 16, 24, 48 h",
            "hj(t+1) = rhoj hj + gainj Qroom",
            "monotone delayed heating effect",
        ),
        "learned",
    ),
    Node(
        "temperature",
        1005,
        590,
        170,
        150,
        "Temperature update",
        (
            "Teq = Tfree + sum(hj)",
            "alpha = NN(metadata) only",
            "bounded equilibrium step",
            "Tin feeds next step",
        ),
        "physics",
    ),
)


COLORS = {
    "persistent": ("#fff1db", "#d97706"),
    "local": ("#ffe8e7", "#d94b44"),
    "input": ("#f2f5f7", "#60717c"),
    "learned": ("#e8f0ff", "#3f6db3"),
    "stochastic": ("#fff5dc", "#b7791f"),
    "physics": ("#e6f5ec", "#31855b"),
    "output": ("#e2f5f5", "#16858b"),
}


def _node_svg(node: Node, *, xi_active: bool) -> str:
    fill, stroke = COLORS[node.kind]
    disabled = node.key in {"initial", "controller", "q_to_t"} and not xi_active
    title = escape(node.title)
    lines = list(node.lines)
    if disabled:
        lines.append("xi input fixed to zero in this variant")
    title_y = node.y + 25
    line_y = node.y + 52
    body = "".join(
        f'<text x="{node.x + 13}" y="{line_y + 20 * index}" class="node-line">'
        f"{escape(line)}</text>"
        for index, line in enumerate(lines)
    )
    badge = ""
    if disabled:
        badge = (
            f'<rect x="{node.x + node.width - 67}" y="{node.y + 8}" width="55" '
            'height="20" rx="3" fill="#eceff1" stroke="#8a969c"/>'
            f'<text x="{node.x + node.width - 39}" y="{node.y + 22}" '
            'class="badge" text-anchor="middle">xi = 0</text>'
        )
    return (
        f'<g class="node {node.kind}"><title>{title}: {escape("; ".join(lines))}</title>'
        f'<rect x="{node.x}" y="{node.y}" width="{node.width}" height="{node.height}" '
        f'rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        f'<text x="{node.x + 13}" y="{title_y}" class="node-title">{title}</text>'
        f"{body}{badge}</g>"
    )


def _path(
    d: str,
    *,
    kind: str = "signal",
    label: str = "",
    label_x: int = 0,
    label_y: int = 0,
    marker: bool = True,
) -> str:
    marker_attr = f' marker-end="url(#arrow-{kind})"' if marker else ""
    label_svg = (
        f'<text x="{label_x}" y="{label_y}" class="edge-label {kind}">{escape(label)}</text>'
        if label
        else ""
    )
    return f'<path d="{d}" class="edge {kind}"{marker_attr}/>{label_svg}'


def _panel(*, legacy: bool) -> str:
    title = "Legacy: shared xi enters four pathways" if legacy else "Thermal-only: xi enters thermal pathway only"
    subtitle = (
        "One persistent draw coherently shifts HP, buffer, Q-to-T and envelope behavior"
        if legacy
        else "HP, initial state and Q-to-T parameters are deterministic given metadata"
    )
    edges = [
        _path("M 228 230 L 270 220", label="c, Tin0", label_x=230, label_y=208),
        _path("M 228 260 L 270 395", label="u(t), Tin(t)", label_x=120, label_y=370),
        _path("M 228 285 C 300 535 330 625 410 650", label="exogenous forcing", label_x=245, label_y=575),
        _path("M 475 220 C 625 220 710 265 800 337", label="E0", label_x=655, label_y=245),
        _path("M 475 235 C 500 370 470 500 500 590", label="x0", label_x=465, label_y=520),
        _path("M 485 410 L 535 410", label="pi, mu, sigma", label_x=488, label_y=397),
        _path("M 640 120 L 640 337", kind="local", label="mode + amplitude draws", label_x=650, label_y=185),
        _path("M 725 410 L 775 410", label="Pel", label_x=738, label_y=397),
        _path("M 965 385 L 1010 385", label="Qroom", label_x=965, label_y=371),
        _path("M 725 380 C 850 275 940 285 1010 365", label="Pel", label_x=870, label_y=286),
        _path("M 870 487 L 850 590", label="Qroom", label_x=864, label_y=545),
        _path("M 680 655 L 1005 655", label="Tfree", label_x=815, label_y=643),
        _path("M 960 655 L 1005 680", label="sum(hj)", label_x=952, label_y=639),
        _path("M 1090 590 L 1090 487", label="Tin", label_x=1098, label_y=540),
        _path("M 850 120 C 800 280 685 430 620 590", kind="local", label="sigma(c,xi,f,x) eps_x", label_x=690, label_y=510),
        _path("M 1085 740 C 1120 790 245 800 245 430 L 270 430", kind="feedback", label="recursive Tin affects thermostat, HP history and thermal forcing", label_x=470, label_y=788),
    ]
    if legacy:
        edges.extend(
            [
                _path("M 125 120 C 145 145 220 170 270 195", kind="persistent", label="x0, E0", label_x=160, label_y=150),
                _path("M 115 120 C 115 300 180 390 270 400", kind="persistent", label="HP probabilities, power, COP, loss", label_x=75, label_y=340),
                _path("M 150 120 C 210 500 570 545 735 625", kind="persistent", label="Q-to-T gains and initial modes", label_x=425, label_y=548),
                _path("M 175 120 C 280 360 310 560 410 625", kind="persistent", label="Mt, bt, sigma, Tfree", label_x=255, label_y=500),
            ]
        )
    else:
        edges.append(
            _path("M 150 120 C 240 350 310 560 410 625", kind="persistent", label="Mt, bt, sigma, Tfree only", label_x=235, label_y=510)
        )
        edges.extend(
            [
                _path("M 125 120 C 145 145 220 170 270 195", kind="disabled", label="zero xi", label_x=160, label_y=150),
                _path("M 115 120 C 115 300 180 390 270 400", kind="disabled", label="zero xi", label_x=125, label_y=345),
                _path("M 150 120 C 210 500 570 545 735 625", kind="disabled", label="zero xi", label_x=500, label_y=555),
            ]
        )
    nodes = "".join(_node_svg(node, xi_active=legacy) for node in NODES)
    return f"""
    <section class="diagram-panel">
      <h2>{escape(title)}</h2>
      <p class="panel-subtitle">{escape(subtitle)}</p>
      <svg viewBox="0 0 1200 820" role="img" aria-label="{escape(title)}">
        <defs>
          <marker id="arrow-signal" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#52616b"/></marker>
          <marker id="arrow-persistent" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#d97706"/></marker>
          <marker id="arrow-local" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#d94b44"/></marker>
          <marker id="arrow-feedback" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#7656a8"/></marker>
          <marker id="arrow-disabled" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#9aa4a9"/></marker>
        </defs>
        {''.join(edges)}
        {nodes}
      </svg>
    </section>
    """


def render() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Structured emulator uncertainty pathways</title>
<style>
  :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #213547; background: #f7f9fa; }}
  body {{ margin: 0; }}
  main {{ max-width: 1800px; margin: 0 auto; padding: 30px 26px 50px; }}
  h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
  .lead {{ max-width: 1100px; margin: 8px 0 20px; color: #50616d; font-size: 16px; line-height: 1.45; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 16px 0 22px; font-size: 14px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
  .swatch {{ width: 25px; height: 5px; border-radius: 2px; display: inline-block; }}
  .persistent-swatch {{ background: #d97706; }} .local-swatch {{ background: #d94b44; }}
  .signal-swatch {{ background: #52616b; }} .feedback-swatch {{ background: #7656a8; }}
  .panels {{ display: grid; grid-template-columns: repeat(2, minmax(660px, 1fr)); gap: 18px; align-items: start; }}
  .diagram-panel {{ background: #fff; border: 1px solid #dce3e7; border-radius: 6px; padding: 18px 14px 10px; overflow-x: auto; }}
  .diagram-panel h2 {{ margin: 0 8px; font-size: 20px; letter-spacing: 0; }}
  .panel-subtitle {{ margin: 5px 8px 4px; color: #63747f; font-size: 14px; }}
  svg {{ width: 100%; min-width: 650px; height: auto; display: block; }}
  .node-title {{ font-weight: 700; font-size: 16px; fill: #1f3440; }}
  .node-line {{ font-size: 12.5px; fill: #354b57; }}
  .badge {{ font-size: 11px; font-weight: 700; fill: #59676e; }}
  .edge {{ fill: none; stroke: #52616b; stroke-width: 2; opacity: .9; }}
  .edge.persistent {{ stroke: #d97706; stroke-width: 3; stroke-dasharray: 8 5; }}
  .edge.local {{ stroke: #d94b44; stroke-width: 2.5; stroke-dasharray: 5 4; }}
  .edge.feedback {{ stroke: #7656a8; stroke-width: 2.5; }}
  .edge.disabled {{ stroke: #9aa4a9; stroke-width: 2; stroke-dasharray: 3 6; opacity: .65; }}
  .edge-label {{ font-size: 11px; fill: #52616b; paint-order: stroke; stroke: #fff; stroke-width: 4px; stroke-linejoin: round; }}
  .edge-label.persistent {{ fill: #a95800; }} .edge-label.local {{ fill: #b6332e; }}
  .edge-label.feedback {{ fill: #654792; }} .edge-label.disabled {{ fill: #7a878d; }}
  .details {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 18px; margin-top: 18px; }}
  .band {{ background: #fff; border: 1px solid #dce3e7; border-radius: 6px; padding: 18px 20px; }}
  .band h2 {{ font-size: 19px; margin: 0 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ text-align: left; vertical-align: top; padding: 9px 10px; border-bottom: 1px solid #e4e9ec; }}
  th {{ background: #f2f5f7; color: #334a56; }}
  .yes {{ color: #24734e; font-weight: 700; }} .no {{ color: #7a878d; }}
  .callout {{ border-left: 4px solid #d97706; padding-left: 13px; line-height: 1.5; color: #3d515d; }}
  .callout + .callout {{ margin-top: 15px; border-color: #31855b; }}
  code {{ background: #edf1f3; border-radius: 3px; padding: 1px 4px; }}
  @media (max-width: 1200px) {{ .panels, .details {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><main>
  <h1>Where uncertainty enters the structured closed-loop HP emulator</h1>
  <p class="lead">The two models have the same networks, physics, local random draws and bounded states. The only architectural difference is which blocks receive the same trajectory-persistent latent draw <strong>xi(k)</strong>. Dashed orange arrows are therefore the controlled ablation.</p>
  <div class="legend">
    <span><i class="swatch persistent-swatch"></i> persistent across the full trajectory</span>
    <span><i class="swatch local-swatch"></i> newly sampled each time step</span>
    <span><i class="swatch signal-swatch"></i> deterministic signal propagation</span>
    <span><i class="swatch feedback-swatch"></i> recursive closed-loop feedback</span>
  </div>
  <div class="panels">{_panel(legacy=True)}{_panel(legacy=False)}</div>
  <div class="details">
    <section class="band">
      <h2>Active uncertainty budget in the matched experiment</h2>
      <table>
        <thead><tr><th>Source</th><th>Persistence</th><th>Legacy</th><th>Thermal-only xi</th><th>Where it acts</th></tr></thead>
        <tbody>
          <tr><td>xi(k), dimension 4</td><td>Whole year</td><td class="yes">Four pathways</td><td class="yes">Thermal only</td><td>Generated parameters and initial conditions</td></tr>
          <tr><td>HP on/off uniform draw</td><td>Every 15 min</td><td class="yes">Scenario inference</td><td class="yes">Scenario inference</td><td>Asymmetric-Markov compressor mode</td></tr>
          <tr><td>Active-power Gaussian draw</td><td>Every 15 min</td><td class="yes">Scenario inference</td><td class="yes">Scenario inference</td><td>Bounded log-normal Pel amplitude</td></tr>
          <tr><td>Thermal eps_x(k,t)</td><td>Every 15 min</td><td class="yes">Learned scale</td><td class="yes">Learned scale</td><td>Bounded latent thermal transition</td></tr>
          <tr><td>Temperature alpha</td><td>Building-specific</td><td class="no">Deterministic</td><td class="no">Deterministic</td><td>Metadata-conditioned temperature update</td></tr>
          <tr><td>Controller-state noise</td><td>Every step</td><td class="no">Disabled</td><td class="no">Disabled</td><td>Configured scale is zero</td></tr>
          <tr><td>Independent Qroom innovation</td><td>Every step</td><td class="no">Disabled</td><td class="no">Disabled</td><td>Qroom is random only through upstream Pel/buffer paths</td></tr>
        </tbody>
      </table>
    </section>
    <section class="band">
      <h2>Why this ablation exposes a trade-off</h2>
      <p class="callout"><strong>Legacy:</strong> the same xi draw changes HP activation and power, COP/loss, initial stored energy, Q-to-T gain, and thermal decay together. That creates physically coherent trajectory-level covariance, but one unfavorable draw can remain a low-heat/high-loss thermal regime for an entire winter and produce conditional overdispersion.</p>
      <p class="callout"><strong>Thermal-only:</strong> persistent covariance in the HP/buffer/Q-to-T chain disappears. Winter Tin bands become much tighter, but the model loses a latent route that was explaining persistent hydronic and intervention-response differences. Training compensates through local HP/process distributions, which need not preserve event-level Qroom behavior.</p>
      <p><strong>Important train/inference distinction:</strong> the matched models were trained with expected HP power, while annual scenarios use Bernoulli mode and bounded log-normal active-power draws. The red uncertainty arrows are active in scenario rollouts; their expected values drive the training thermal path.</p>
    </section>
  </div>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/neural_building_emulator_refactor/"
            "persistent_xi_thermal_only_200_comparison/"
            "uncertainty_pathways_legacy_vs_thermal_only.html"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render())
    print(f"saved_uncertainty_pathway_diagram={args.output}")


if __name__ == "__main__":
    main()
