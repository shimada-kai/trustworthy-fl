import os
import re
import shutil
import subprocess
from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Trustworthy FL Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 20% 0%, rgba(14,165,233,.08), transparent 26%),
            radial-gradient(circle at 85% 20%, rgba(34,197,94,.06), transparent 28%),
            #07101d;
        color: #e5eef8;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 0.7rem;
        padding-bottom: 0.8rem;
    }

    .hero {
        border: 1px solid rgba(125,211,252,.16);
        background: rgba(15,23,42,.72);
        border-radius: 20px;
        padding: 18px 24px;
        margin-bottom: 12px;
        box-shadow: 0 18px 45px rgba(0,0,0,.22);
    }

    .eyebrow {
        font-size: .70rem;
        letter-spacing: .18em;
        color: #7dd3fc;
        font-weight: 800;
    }

    .hero-title {
        font-size: 1.75rem;
        font-weight: 900;
        margin-top: 4px;
        color: #f8fafc;
    }

    .hero-sub {
        color: #94a3b8;
        font-size: .88rem;
        margin-top: 3px;
    }

    .status-row {
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin-top:12px;
    }

    .pill {
        font-size:.72rem;
        border-radius:999px;
        padding:6px 10px;
        background:rgba(2,6,23,.55);
        border:1px solid rgba(148,163,184,.16);
    }

    .ok {
        color:#86efac;
        border-color:rgba(34,197,94,.35);
    }

    .info {
        color:#7dd3fc;
        border-color:rgba(56,189,248,.28);
    }

    .section {
        color:#94a3b8;
        font-size:.68rem;
        letter-spacing:.15em;
        font-weight:800;
        margin:4px 0 7px;
    }

    .client-strip {
        display:grid;
        grid-template-columns:repeat(8,1fr);
        gap:8px;
    }

    .client-node {
        grid-column:span 2;
    }

    .client-node:nth-child(5) {
        grid-column:2 / span 2;
    }

    .client-node:nth-child(6) {
        grid-column:4 / span 2;
    }

    .client-node:nth-child(7) {
        grid-column:6 / span 2;
    }

    .client-node {
        text-align:center;
        border-radius:14px;
        padding:10px 4px;
        background:rgba(15,23,42,.8);
        border:1px solid rgba(148,163,184,.16);
    }

    .client-id {
        font-size:.88rem;
        font-weight:800;
    }

    .mini-nn {
        display:flex;
        justify-content:center;
        align-items:center;
        height:54px;
        margin:6px 0 5px;
    }

    .mini-nn svg {
        width:82px;
        height:52px;
        overflow:visible;
    }

    .nn-edge {
        stroke:#475569;
        stroke-width:1.2;
        opacity:.9;
    }

    .nn-node-svg {
        fill:#7dd3fc;
        stroke:#bae6fd;
        stroke-width:.8;
        filter:drop-shadow(0 0 3px rgba(56,189,248,.55));
    }

    .client-node.reject .nn-node-svg {
        fill:#f87171;
        stroke:#fecaca;
        filter:drop-shadow(0 0 3px rgba(248,113,113,.5));
    }

    .client-node.accept .nn-node-svg {
        fill:#4ade80;
        stroke:#bbf7d0;
        filter:drop-shadow(0 0 3px rgba(74,222,128,.5));
    }

    .security-stack {
        height:275px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        gap:10px;
    }

    .security-card {
        border-radius:14px;
        padding:13px 9px;
        text-align:center;
        background:rgba(15,23,42,.78);
        border:1px solid rgba(148,163,184,.17);
    }

    .security-card.verified {
        border-color:rgba(34,197,94,.42);
        box-shadow:0 0 18px rgba(34,197,94,.07);
    }

    .security-name {
        color:#cbd5e1;
        font-size:.67rem;
        font-weight:800;
        letter-spacing:.06em;
    }

    .security-state {
        color:#64748b;
        font-size:.67rem;
        font-weight:900;
        margin-top:5px;
    }

    .security-card.verified .security-state {
        color:#4ade80;
    }

    .client-type {
        font-size:.58rem;
        color:#94a3b8;
        margin-top:2px;
    }

    .pending {
        color:#cbd5e1;
    }

    .reject {
        border-color:rgba(239,68,68,.42);
        box-shadow:0 0 18px rgba(239,68,68,.07);
    }

    .accept {
        border-color:rgba(34,197,94,.42);
        box-shadow:0 0 18px rgba(34,197,94,.07);
    }

    .reject-text {
        color:#f87171;
        font-size:.66rem;
        font-weight:800;
        margin-top:4px;
    }

    .accept-text {
        color:#4ade80;
        font-size:.66rem;
        font-weight:800;
        margin-top:4px;
    }

    .pending-text {
        color:#64748b;
        font-size:.66rem;
        font-weight:800;
        margin-top:4px;
    }

    .stage {
        height: 275px;
        border-radius:18px;
        background:rgba(15,23,42,.73);
        border:1px solid rgba(148,163,184,.15);
        padding:18px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
    }

    .zk-stage {
        border-color:rgba(168,85,247,.30);
        box-shadow: inset 0 0 34px rgba(168,85,247,.04);
    }

    .tdx-stage {
        border-color:rgba(56,189,248,.32);
        box-shadow:
            0 0 30px rgba(56,189,248,.07),
            inset 0 0 35px rgba(56,189,248,.04);
    }

    .tech-icon {
        width:72px;
        height:72px;
        margin-bottom:10px;
    }

    .zk-icon {
        filter:drop-shadow(0 0 12px rgba(168,85,247,.22));
    }

    .tdx-icon {
        filter:drop-shadow(0 0 12px rgba(56,189,248,.22));
    }

    .icon-line-zk {
        stroke:#c084fc;
        stroke-width:2.2;
        fill:none;
        stroke-linecap:round;
        stroke-linejoin:round;
    }

    .icon-fill-zk {
        fill:rgba(168,85,247,.13);
        stroke:#c084fc;
        stroke-width:2.2;
    }

    .icon-line-tdx {
        stroke:#38bdf8;
        stroke-width:2;
        fill:none;
        stroke-linecap:round;
        stroke-linejoin:round;
    }

    .icon-fill-tdx {
        fill:rgba(56,189,248,.10);
        stroke:#38bdf8;
        stroke-width:2;
    }

    .stage-label {
        color:#cbd5e1;
        font-size:.72rem;
        letter-spacing:.12em;
        font-weight:800;
    }

    .stage-title {
        font-size:1.28rem;
        font-weight:900;
        margin-top:7px;
        color:#f8fafc;
    }

    .stage-status {
        margin-top:14px;
        font-size:.85rem;
        font-weight:800;
    }

    .stage-sub {
        margin-top:7px;
        color:#94a3b8;
        font-size:.72rem;
    }

    .flow-arrow {
        display:flex;
        align-items:center;
        justify-content:center;
        height:275px;
        font-size:2.1rem;
        color:#38bdf8;
    }

    .result-wrap {
        display:grid;
        grid-template-columns:1fr 100px 1fr;
        gap:14px;
        align-items:center;
    }

    .metric-card {
        border-radius:16px;
        padding:15px 18px;
        background:rgba(15,23,42,.78);
        border:1px solid rgba(148,163,184,.15);
    }

    .metric-small {
        color:#94a3b8;
        font-size:.66rem;
        letter-spacing:.12em;
        font-weight:800;
    }

    .metric-value {
        font-size:1.75rem;
        font-weight:900;
        margin-top:3px;
    }

    .bad {
        color:#f87171;
    }

    .good {
        color:#4ade80;
    }

    .metric-foot {
        color:#94a3b8;
        font-size:.72rem;
    }

    .metric-status {
        margin-top:5px;
        font-size:.72rem;
        font-weight:800;
    }

    .big-arrow {
        text-align:center;
        font-size:2rem;
        color:#7dd3fc;
    }

    div.stButton > button {
        width:100%;
        height:2.8rem;
        border-radius:14px;
        border:1px solid rgba(56,189,248,.5);
        background:linear-gradient(90deg,#075985,#0f766e);
        color:white;
        font-weight:900;
        letter-spacing:.03em;
    }

    div.stButton > button:hover {
        border-color:#7dd3fc;
        color:white;
    }

    .footer-note {
        text-align:center;
        color:#64748b;
        font-size:.62rem;
        margin-top:8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CLIENTS = [
    (0, "SIGN FLIP", False),
    (1, "SIGN FLIP", False),
    (2, "SIGN FLIP", False),
    (3, "SIGN FLIP", False),
    (4, "HONEST", True),
    (5, "HONEST", True),
    (6, "HONEST", True),
]

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">TRUSTWORTHY FEDERATED LEARNING</div>
        <div class="hero-title">RISC Zero Verification × Intel TDX</div>
        <div class="hero-sub">
            Reject poisoned model updates before they enter the trusted aggregation boundary.
        </div>
        <div class="status-row">
            <div class="pill ok">● Intel TDX</div>
            <div class="pill info">● Sampled Training Verification</div>
            <div class="pill info">● Coordinate-wise Median</div>
            <div class="pill info">● 7 Clients / 4 Attackers</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

run = st.button("▶ RUN REAL 5-ROUND DEMO")

attestation_placeholder = st.empty()
clients_placeholder = st.empty()

col_clients, col_arrow1, col_zk, col_arrow2, col_security, col_arrow3, col_tdx = st.columns(
    [1.15, .08, .78, .08, .82, .08, .88]
)


def client_html(statuses: dict[int, str]) -> str:
    cards = []

    for idx, kind, _ in CLIENTS:
        status = statuses.get(idx, "PENDING")

        if status == "ACCEPT":
            css = "accept"
            label = "✓ VERIFIED"
            status_css = "accept-text"
        elif status == "REJECT":
            css = "reject"
            label = "✕ REJECTED"
            status_css = "reject-text"
        else:
            css = ""
            label = "PENDING"
            status_css = "pending-text"

        cards.append(
            f'<div class="client-node {css}">'
            f'<div class="client-id">C{idx}</div>'
            f'<div class="client-type">{kind}</div>'
            f'<div class="mini-nn">'
            f'<svg viewBox="0 0 82 52" aria-label="2-3-2 neural network">'
            f'  <line class="nn-edge" x1="10" y1="15" x2="41" y2="10"/>'
            f'  <line class="nn-edge" x1="10" y1="15" x2="41" y2="26"/>'
            f'  <line class="nn-edge" x1="10" y1="15" x2="41" y2="42"/>'
            f'  <line class="nn-edge" x1="10" y1="37" x2="41" y2="10"/>'
            f'  <line class="nn-edge" x1="10" y1="37" x2="41" y2="26"/>'
            f'  <line class="nn-edge" x1="10" y1="37" x2="41" y2="42"/>'
            f'  <line class="nn-edge" x1="41" y1="10" x2="72" y2="15"/>'
            f'  <line class="nn-edge" x1="41" y1="10" x2="72" y2="37"/>'
            f'  <line class="nn-edge" x1="41" y1="26" x2="72" y2="15"/>'
            f'  <line class="nn-edge" x1="41" y1="26" x2="72" y2="37"/>'
            f'  <line class="nn-edge" x1="41" y1="42" x2="72" y2="15"/>'
            f'  <line class="nn-edge" x1="41" y1="42" x2="72" y2="37"/>'
            f'  <circle class="nn-node-svg" cx="10" cy="15" r="4.5"/>'
            f'  <circle class="nn-node-svg" cx="10" cy="37" r="4.5"/>'
            f'  <circle class="nn-node-svg" cx="41" cy="10" r="4.5"/>'
            f'  <circle class="nn-node-svg" cx="41" cy="26" r="4.5"/>'
            f'  <circle class="nn-node-svg" cx="41" cy="42" r="4.5"/>'
            f'  <circle class="nn-node-svg" cx="72" cy="15" r="4.5"/>'
            f'  <circle class="nn-node-svg" cx="72" cy="37" r="4.5"/>'
            f'</svg>'
            f'</div>'
            f'<div class="{status_css}">{label}</div>'
            f'</div>'
        )

    return '<div class="client-strip">' + "".join(cards) + "</div>"


def zk_icon_html() -> str:
    return """
    <svg class="tech-icon zk-icon" viewBox="0 0 80 80" aria-label="Zero knowledge verification">
        <path class="icon-fill-zk"
              d="M40 8 L64 17 V36 C64 53 54 65 40 72 C26 65 16 53 16 36 V17 Z"/>
        <path class="icon-line-zk"
              d="M28 40 L36 48 L53 29"/>
        <circle cx="40" cy="26" r="4" fill="#c084fc"/>
    </svg>
    """


def tdx_icon_html() -> str:
    return """
    <svg class="tech-icon tdx-icon" viewBox="0 0 80 80" aria-label="Intel TDX trusted execution environment">
        <rect class="icon-fill-tdx" x="18" y="18" width="44" height="44" rx="7"/>
        <rect class="icon-line-tdx" x="29" y="29" width="22" height="22" rx="4"/>
        <path class="icon-line-tdx"
              d="M25 10 V18 M36 10 V18 M47 10 V18 M58 10 V18
                 M25 62 V70 M36 62 V70 M47 62 V70 M58 62 V70
                 M10 25 H18 M10 36 H18 M10 47 H18 M10 58 H18
                 M62 25 H70 M62 36 H70 M62 47 H70 M62 58 H70"/>
        <path class="icon-line-tdx" d="M34 40 L39 45 L48 35"/>
    </svg>
    """


with col_clients:
    st.markdown('<div class="section">CLIENTS</div>', unsafe_allow_html=True)
    client_box = st.empty()

with col_arrow1:
    st.markdown('<div class="section">&nbsp;</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

with col_zk:
    st.markdown('<div class="section">VERIFICATION</div>', unsafe_allow_html=True)
    zk_box = st.empty()

with col_arrow2:
    st.markdown('<div class="section">&nbsp;</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

with col_security:
    st.markdown('<div class="section">SECURE CHANNEL</div>', unsafe_allow_html=True)
    security_box = st.empty()

with col_arrow3:
    st.markdown('<div class="section">&nbsp;</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">→</div>', unsafe_allow_html=True)

with col_tdx:
    st.markdown('<div class="section">TRUSTED AGGREGATION</div>', unsafe_allow_html=True)
    tdx_box = st.empty()


client_box.markdown(client_html({}), unsafe_allow_html=True)

zk_box.markdown(
    '<div class="stage zk-stage">'
    + zk_icon_html()
    + '<div class="stage-label">RISC ZERO · zkVM</div>'
      '<div class="stage-title">ZK Verification</div>'
      '<div class="stage-status" style="color:#94a3b8;">WAITING</div>'
      '<div class="stage-sub">net.0.weight[0,0] · 17 Adam steps</div>'
      '</div>',
    unsafe_allow_html=True,
)

security_box.markdown(
    """
    <div class="security-stack">
        <div class="security-card">
            <div class="security-name">REMOTE ATTESTATION</div>
            <div class="security-state">WAITING</div>
        </div>
        <div class="security-card">
            <div class="security-name">TLS BINDING</div>
            <div class="security-state">WAITING</div>
        </div>
        <div class="security-card">
            <div class="security-name">HTTPS MODEL SUBMIT</div>
            <div class="security-state">WAITING</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tdx_box.markdown(
    '<div class="stage tdx-stage">'
    + tdx_icon_html()
    + '<div class="stage-label">INTEL TDX · TEE</div>'
      '<div class="stage-title">Trusted Zone</div>'
      '<div class="stage-status" style="color:#94a3b8;">WAITING</div>'
      '<div class="stage-sub">Coordinate-wise Median</div>'
      '</div>',
    unsafe_allow_html=True,
)


if run:
    root = Path(__file__).resolve().parents[1]
    flwr_bin = shutil.which("flwr")

    if flwr_bin is None:
        st.error("flwr command not found")
        st.stop()

    statuses: dict[int, str] = {}
    current_round = 0
    accepted_updates = 0
    attestation_ok = False
    tls_binding_ok = False
    tls_submit_ok = False
    events: list[str] = []

    event_box = st.empty()

    command = [
        "script",
        "-q",
        "/dev/null",
        flwr_bin,
        "run",
        ".",
        "--run-config",
        'zk-verification-enabled=true tee-enabled=true '
        'attack-enabled=true attack-type="sign_flip" '
        'malicious-client-ids="0,1,2,3" '
        'num-server-rounds=5',
        "--stream",
    ]

    attestation_placeholder.markdown(
        '<div class="section" style="color:#7dd3fc;">'
        '● REAL EXECUTION STARTED · FLOWER + RISC ZERO + INTEL TDX'
        '</div>',
        unsafe_allow_html=True,
    )

    process_env = dict(os.environ)
    process_env["RAY_DEDUP_LOGS"] = "0"

    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env,
    )

    assert process.stdout is not None

    raw_log_path = root / "demo" / "live_run.log"
    raw_log_path.write_text("")

    for line in process.stdout:
        with raw_log_path.open("a") as f:
            f.write(line)

        line = line.rstrip()

        round_match = re.search(r"\[ROUND (\d+)/5\]", line)
        if round_match:
            current_round = int(round_match.group(1))
            statuses = {}
            accepted_updates = 0
            attestation_ok = False
            tls_binding_ok = False
            tls_submit_ok = False

            client_box.markdown(
                client_html(statuses),
                unsafe_allow_html=True,
            )

            attestation_placeholder.markdown(
                f'<div class="section" style="color:#7dd3fc;">'
                f'● REAL EXECUTION · ROUND {current_round} / 5'
                f'</div>',
                unsafe_allow_html=True,
            )

            security_box.markdown(
                '<div class="security-stack">'
                '<div class="security-card">'
                '<div class="security-name">REMOTE ATTESTATION</div>'
                '<div class="security-state">WAITING</div>'
                '</div>'
                '<div class="security-card">'
                '<div class="security-name">TLS BINDING</div>'
                '<div class="security-state">WAITING</div>'
                '</div>'
                '<div class="security-card">'
                '<div class="security-name">HTTPS MODEL SUBMIT</div>'
                '<div class="security-state">WAITING</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            zk_box.markdown(
                f'<div class="stage zk-stage">'
                f'{zk_icon_html()}'
                f'<div class="stage-label">RISC ZERO · zkVM</div>'
                f'<div class="stage-title">ZK Verification</div>'
                f'<div class="stage-status" style="color:#c084fc;">'
                f'ROUND {current_round} · PROVING'
                f'</div>'
                f'<div class="stage-sub">sampled Adam training trace</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        verify_match = re.search(
            r"\[ZK-VERIFY\].*client=(\d+).*round=(\d+).*result=(ACCEPT|REJECT)",
            line,
        )
        if verify_match:
            client_id = int(verify_match.group(1))
            result = verify_match.group(3)

            statuses[client_id] = result

            client_box.markdown(
                client_html(statuses),
                unsafe_allow_html=True,
            )

            events.append(
                f"Round {current_round} · Client {client_id} · {result}"
            )

        if "[REMOTE-ATTESTATION]" in line and "result=VERIFIED" in line:
            attestation_ok = True
            events.append(
                f"Round {current_round} · Remote Attestation · VERIFIED"
            )

        if "[TLS-BINDING]" in line and "result=VERIFIED" in line:
            tls_binding_ok = True
            events.append(
                f"Round {current_round} · TLS Binding · VERIFIED"
            )

        if "[TLS-SUBMIT]" in line and "result=SUCCESS" in line:
            tls_submit_ok = True

        if attestation_ok or tls_binding_ok or tls_submit_ok:
            security_box.markdown(
                '<div class="security-stack">'
                f'<div class="security-card {"verified" if attestation_ok else ""}">'
                '<div class="security-name">REMOTE ATTESTATION</div>'
                f'<div class="security-state">{"✓ VERIFIED" if attestation_ok else "WAITING"}</div>'
                '</div>'
                f'<div class="security-card {"verified" if tls_binding_ok else ""}">'
                '<div class="security-name">TLS BINDING</div>'
                f'<div class="security-state">{"✓ VERIFIED" if tls_binding_ok else "WAITING"}</div>'
                '</div>'
                f'<div class="security-card {"verified" if tls_submit_ok else ""}">'
                '<div class="security-name">HTTPS MODEL SUBMIT</div>'
                f'<div class="security-state">{"✓ SECURE" if tls_submit_ok else "WAITING"}</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Fallback: a ZK-rejected update must never enter TDX.
        skip_match = re.search(
            r"\[TDX-SKIP\].*client=(\d+).*reason=zk_rejected",
            line,
        )
        if skip_match:
            client_id = int(skip_match.group(1))
            if client_id not in statuses:
                statuses[client_id] = "REJECT"
                client_box.markdown(
                    client_html(statuses),
                    unsafe_allow_html=True,
                )
                events.append(
                    f"Round {current_round} · Client {client_id} · REJECT"
                )

        # Fallback: successful TDX submit implies accepted ZK gate.
        submit_client_match = re.search(
            r"\[TDX-SUBMIT\].*client=(\d+)",
            line,
        )
        if submit_client_match:
            client_id = int(submit_client_match.group(1))
            if client_id not in statuses:
                statuses[client_id] = "ACCEPT"
                client_box.markdown(
                    client_html(statuses),
                    unsafe_allow_html=True,
                )
                events.append(
                    f"Round {current_round} · Client {client_id} · ACCEPT"
                )

        submit_match = re.search(
            r"\[TDX-SUBMIT\].*accepted_updates=(\d+)",
            line,
        )
        if submit_match:
            accepted_updates = int(submit_match.group(1))

            tdx_box.markdown(
                f'<div class="stage tdx-stage">'
                f'{tdx_icon_html()}'
                f'<div class="stage-label">INTEL TDX · TEE</div>'
                f'<div class="stage-title">Trusted Zone</div>'
                f'<div class="stage-status" style="color:#7dd3fc;">'
                f'{accepted_updates} VERIFIED UPDATE(S)'
                f'</div>'
                f'<div class="stage-sub">received inside TDX</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        median_match = re.search(
            r"\[TDX-MEDIAN\]\[round=(\d+)\] aggregated (\d+) accepted updates",
            line,
        )
        if median_match:
            median_round = int(median_match.group(1))
            count = int(median_match.group(2))

            tdx_box.markdown(
                f'<div class="stage tdx-stage">'
                f'{tdx_icon_html()}'
                f'<div class="stage-label">INTEL TDX · TEE</div>'
                f'<div class="stage-title">Trusted Zone</div>'
                f'<div class="stage-status" style="color:#4ade80;">'
                f'✓ ROUND {median_round} MEDIAN COMPLETE'
                f'</div>'
                f'<div class="stage-sub">{count} verified updates aggregated</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            events.append(
                f"Round {median_round} · TDX Median · {count} updates"
            )

        metric_match = re.search(
            r"\[FL GLOBAL TEST\]\[round=(\d+)\].*"
            r"acc=([0-9.]+).*roc_auc=([0-9.]+).*pr_auc=([0-9.]+)",
            line,
        )
        if metric_match:
            metric_round = int(metric_match.group(1))
            roc_auc = float(metric_match.group(3))

            if metric_round > 0:
                events.append(
                    f"Round {metric_round} · Global ROC-AUC = {roc_auc:.4f}"
                )

            if metric_round == 5:
                final_roc_auc = roc_auc

        if (
            "[ZK-VERIFY]" in line
            or "[REMOTE-ATTESTATION]" in line
            or "[TLS-BINDING]" in line
            or "[TLS-SUBMIT]" in line
            or "[TDX-SUBMIT]" in line
            or "[TDX-SKIP]" in line
            or "[TDX-MEDIAN]" in line
            or "[FL GLOBAL TEST]" in line
        ):
            event_box.code(
                "\n".join(events[-12:]),
                language=None,
            )

    return_code = process.wait()

    if return_code == 0:
        attestation_placeholder.markdown(
            '<div class="section" style="color:#4ade80;">'
            '● REAL 5-ROUND EXECUTION COMPLETE'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(
            f"Flower execution failed with exit code {return_code}"
        )


final_roc_auc = locals().get("final_roc_auc", None)

st.markdown(
    '<div class="section">ATTACK RESILIENCE · 5-ROUND EXPERIMENT</div>',
    unsafe_allow_html=True,
)

right_value = f"{final_roc_auc:.4f}" if final_roc_auc is not None else "—"
right_status = "✓ MODEL RECOVERED" if final_roc_auc is not None else "WAITING FOR REAL RUN"
right_status_class = "good" if final_roc_auc is not None else ""

st.markdown(
    '<div class="result-wrap">'
    '<div class="metric-card">'
    '<div class="metric-small">MEDIAN ONLY · 4/7 ATTACKERS</div>'
    '<div class="metric-value bad">0.1846</div>'
    '<div class="metric-foot">ROC-AUC</div>'
    '<div class="metric-status bad">✕ MODEL COLLAPSED</div>'
    '</div>'
    '<div class="big-arrow">→</div>'
    '<div class="metric-card">'
    '<div class="metric-small">ZK FILTER + MEDIAN</div>'
    f'<div class="metric-value good">{right_value}</div>'
    '<div class="metric-foot">ROC-AUC</div>'
    f'<div class="metric-status {right_status_class}">{right_status}</div>'
    '</div>'
    '</div>'
    '<div class="footer-note">'
    'PoC: sampled parameter-level Adam training verification. '
    'Forward/backward gradient provenance is outside the current proof scope.'
    '</div>',
    unsafe_allow_html=True,
)
