import json
import io
import html
import contextlib
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
PLOTS_DIR = SRC_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

orig_df = pd.read_csv(SRC_DIR / "original_features_combined.csv")
merged_df = pd.read_csv(SRC_DIR / "merged_robustness_dataset.csv", low_memory=False)


def svg_header(width: int, height: int, bg: str = "#0f172a") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}" style="background:{bg}; font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; border-radius: 12px; border: 1px solid #334155;">',
        '<defs>',
        '  <linearGradient id="gradRed" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%" stop-color="#ef4444" stop-opacity="0.85"/>',
        '    <stop offset="100%" stop-color="#f97316" stop-opacity="0.95"/>',
        '  </linearGradient>',
        '  <linearGradient id="gradBlue" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.85"/>',
        '    <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.95"/>',
        '  </linearGradient>',
        '  <linearGradient id="gradGreen" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%" stop-color="#10b981" stop-opacity="0.85"/>',
        '    <stop offset="100%" stop-color="#34d399" stop-opacity="0.95"/>',
        '  </linearGradient>',
        '  <linearGradient id="gradPurple" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%" stop-color="#8b5cf6" stop-opacity="0.85"/>',
        '    <stop offset="100%" stop-color="#ec4899" stop-opacity="0.95"/>',
        '  </linearGradient>',
        '</defs>',
    ]


def make_fig1_attack_ranking(df: pd.DataFrame) -> str:
    """Figure 1: Dual-panel Attack Ranking (Mean Z-Drop + Pareto Frontier vs Semantic Similarity)."""
    att = df.groupby(["attack_name", "intensity"]).agg(
        mean_z_drop=("z_score_drop", "mean"),
        rel_drop=("relative_z_drop_pct", "mean"),
        survival=("watermark_survived", "mean"),
        sem_sim=("semantic_similarity_cosine", "mean"),
    ).reset_index()
    att["label"] = att["attack_name"] + " (" + att["intensity"] + ")"
    att = att.sort_values("mean_z_drop", ascending=False).reset_index(drop=True)

    W, H = 1100, 650
    lines = svg_header(W, H)

    lines.append('<text x="32" y="38" fill="#f8fafc" font-size="20" font-weight="700">Figure 1: Attack Effectiveness Ranking &amp; Semantic Preservation Trade-off</text>')
    lines.append('<text x="32" y="60" fill="#94a3b8" font-size="13">Left: Mean KGW Z-Score Drop across 600 documents (initial mean Z = 14.79) | Right: Z-Score Drop (%) vs. Cosine Semantic Similarity</text>')

    left_x, right_x = 215, 555
    top_y, row_h = 95, 28
    max_drop = 15.0

    lines.append(f'<text x="{left_x}" y="82" fill="#cbd5e1" font-size="12" font-weight="600">Mean Z-Score Drop (ΔZ) &amp; Post-Attack Survival Rate</text>')
    for tick in [0, 3, 6, 9, 12, 15]:
        tx = left_x + (tick / max_drop) * (right_x - left_x)
        lines.append(f'<line x1="{tx:.1f}" y1="{top_y - 5}" x2="{tx:.1f}" y2="{top_y + 18 * row_h}" stroke="#1e293b" stroke-width="1"/>')
        lines.append(f'<text x="{tx:.1f}" y="{top_y + 18 * row_h + 18}" fill="#64748b" font-size="11" text-anchor="middle">{tick}</text>')

    for idx, row in att.iterrows():
        cy = top_y + idx * row_h
        bw = max(2.0, (row["mean_z_drop"] / max_drop) * (right_x - left_x))
        fill = "url(#gradRed)" if row["attack_name"] == "humarin" else ("url(#gradPurple)" if row["mean_z_drop"] > 4.0 else "url(#gradBlue)")
        lines.append(f'<text x="{left_x - 10}" y="{cy + 15}" fill="#e2e8f0" font-size="11.5" text-anchor="end">{html.escape(row["label"])}</text>')
        lines.append(f'<rect x="{left_x}" y="{cy + 3}" width="{bw:.1f}" height="18" rx="4" fill="{fill}"/>')
        surv_pct = row["survival"] * 100
        surv_color = "#f87171" if surv_pct < 50 else ("#fbbf24" if surv_pct < 90 else "#34d399")
        lines.append(f'<text x="{left_x + bw + 6:.1f}" y="{cy + 16}" fill="#f1f5f9" font-size="11" font-weight="600">-{row["mean_z_drop"]:.2f} <tspan fill="{surv_color}">({surv_pct:.1f}% surv)</tspan></text>')

    px0, px1 = 680, 1055
    py0, py1 = 95, 595
    lines.append(f'<text x="{px0}" y="82" fill="#cbd5e1" font-size="12" font-weight="600">Pareto Frontier: Semantic Similarity vs. Relative Z-Drop (%)</text>')
    lines.append(f'<rect x="{px0}" y="{py0}" width="{px1 - px0}" height="{py1 - py0}" fill="#1e293b" fill-opacity="0.45" stroke="#334155" rx="8"/>')

    xmin, xmax = 0.82, 1.005
    ymin, ymax = -2.0, 100.0

    for xtick in [0.84, 0.88, 0.92, 0.96, 1.00]:
        xx = px0 + ((xtick - xmin) / (xmax - xmin)) * (px1 - px0)
        lines.append(f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py1}" stroke="#334155" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{xx:.1f}" y="{py1 + 18}" fill="#94a3b8" font-size="11" text-anchor="middle">{xtick:.2f}</text>')

    for ytick in [0, 20, 40, 60, 80, 100]:
        yy = py1 - ((ytick - ymin) / (ymax - ymin)) * (py1 - py0)
        lines.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#334155" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{px0 - 8}" y="{yy + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{ytick}%</text>')

    lines.append(f'<text x="{(px0 + px1)/2:.1f}" y="{py1 + 38}" fill="#cbd5e1" font-size="12" text-anchor="middle">Semantic Similarity (Cosine)</text>')

    palette = {
        "humarin": "#ef4444",
        "helm_typo": "#f97316",
        "copy_paste": "#eab308",
        "synonym_wordnet": "#a855f7",
        "helm_lowercase": "#06b6d4",
        "helm_misspelling": "#3b82f6",
        "helm_contraction": "#10b981",
        "helm_expansion": "#10b981",
    }

    for idx, row in att.iterrows():
        xx = px0 + ((row["sem_sim"] - xmin) / (xmax - xmin)) * (px1 - px0)
        yy = py1 - ((row["rel_drop"] - ymin) / (ymax - ymin)) * (py1 - py0)
        col = palette.get(row["attack_name"], "#38bdf8")
        lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6.5" fill="{col}" stroke="#0f172a" stroke-width="1.5"/>')
        if row["rel_drop"] > 14 or row["attack_name"] == "helm_lowercase":
            anchor = "start" if xx < 930 else "end"
            dx = 9 if anchor == "start" else -9
            dy = -6 if row["intensity"] == "high" else (12 if row["intensity"] == "low" else 4)
            lines.append(f'<text x="{xx + dx:.1f}" y="{yy + dy:.1f}" fill="#e2e8f0" font-size="10.5" text-anchor="{anchor}">{html.escape(row["label"])}</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "01_attack_pareto_ranking.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig2_model_domain_comparison(orig: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Figure 2: Model & Domain Comparison (Initial vs Post-Attack Z-Score & Survival Rates)."""
    W, H = 1100, 480
    lines = svg_header(W, H)

    lines.append('<text x="32" y="38" fill="#f8fafc" font-size="20" font-weight="700">Figure 2: Model &amp; Domain Robustness Comparison (Initial vs. Post-Attack Z-Scores)</text>')
    lines.append('<text x="32" y="60" fill="#94a3b8" font-size="13">Left: LLM Comparison (Qwen2.5-1.5B vs. SmolLM3-3B vs. Llama-3.2-1B) | Right: Writing Domain Comparison (600 docs, 10,800 attacks)</text>')

    m_stats = merged.groupby("model_name").agg(
        init_z=("initial_z_score", "mean"),
        att_z=("baseline_z_score", "mean"),
        surv=("watermark_survived", "mean"),
        surv_no_para=("watermark_survived", lambda s: s[merged.loc[s.index, "attack_name"] != "humarin"].mean()),
    ).reindex(["Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-1B-Instruct"])

    short_models = {
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
        "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
        "meta-llama/Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    }

    lx0, lx1, ly0, ly1 = 75, 515, 105, 395
    max_z = 22.0
    lines.append(f'<text x="{lx0}" y="88" fill="#cbd5e1" font-size="12.5" font-weight="600">By Model: Initial Z (Blue) vs. Post-Attack Z (Green)</text>')
    lines.append(f'<rect x="{lx0}" y="{ly0}" width="{lx1 - lx0}" height="{ly1 - ly0}" fill="#1e293b" fill-opacity="0.4" stroke="#334155" rx="8"/>')

    z4_y = ly1 - (4.0 / max_z) * (ly1 - ly0)
    lines.append(f'<line x1="{lx0}" y1="{z4_y:.1f}" x2="{lx1}" y2="{z4_y:.1f}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5,4"/>')
    lines.append(f'<text x="{lx1 - 8}" y="{z4_y - 6:.1f}" fill="#f87171" font-size="10.5" text-anchor="end">Detection Threshold (Z = 4.0)</text>')

    for zt in [0, 5, 10, 15, 20]:
        zy = ly1 - (zt / max_z) * (ly1 - ly0)
        lines.append(f'<line x1="{lx0}" y1="{zy:.1f}" x2="{lx1}" y2="{zy:.1f}" stroke="#334155" stroke-opacity="0.5"/>')
        lines.append(f'<text x="{lx0 - 8}" y="{zy + 4:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{zt}</text>')

    for i, (mname, row) in enumerate(m_stats.iterrows()):
        cx = lx0 + 80 + i * 140
        h_init = (row["init_z"] / max_z) * (ly1 - ly0)
        h_att = (row["att_z"] / max_z) * (ly1 - ly0)

        lines.append(f'<rect x="{cx - 34}" y="{ly1 - h_init:.1f}" width="30" height="{h_init:.1f}" rx="4" fill="url(#gradBlue)"/>')
        lines.append(f'<text x="{cx - 19}" y="{ly1 - h_init - 6:.1f}" fill="#93c5fd" font-size="11" font-weight="600" text-anchor="middle">{row["init_z"]:.1f}</text>')

        lines.append(f'<rect x="{cx + 4}" y="{ly1 - h_att:.1f}" width="30" height="{h_att:.1f}" rx="4" fill="url(#gradGreen)"/>')
        lines.append(f'<text x="{cx + 19}" y="{ly1 - h_att - 6:.1f}" fill="#6ee7b7" font-size="11" font-weight="600" text-anchor="middle">{row["att_z"]:.1f}</text>')

        lines.append(f'<text x="{cx}" y="{ly1 + 22}" fill="#f1f5f9" font-size="12" font-weight="600" text-anchor="middle">{short_models[mname]}</text>')
        lines.append(f'<text x="{cx}" y="{ly1 + 40}" fill="#34d399" font-size="11" text-anchor="middle">{row["surv"]*100:.1f}% surv ({row["surv_no_para"]*100:.1f}% no-para)</text>')

    d_stats = merged.groupby("domain").agg(
        init_z=("initial_z_score", "mean"),
        att_z=("baseline_z_score", "mean"),
        surv=("watermark_survived", "mean"),
    ).reindex(["creative", "conversational", "instructional", "academic"])

    rx0, rx1 = 595, 1065
    lines.append(f'<text x="{rx0}" y="88" fill="#cbd5e1" font-size="12.5" font-weight="600">By Domain: Initial Z (Blue) vs. Post-Attack Z (Green)</text>')
    lines.append(f'<rect x="{rx0}" y="{ly0}" width="{rx1 - rx0}" height="{ly1 - ly0}" fill="#1e293b" fill-opacity="0.4" stroke="#334155" rx="8"/>')
    lines.append(f'<line x1="{rx0}" y1="{z4_y:.1f}" x2="{rx1}" y2="{z4_y:.1f}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5,4"/>')

    for i, (dname, row) in enumerate(d_stats.iterrows()):
        cx = rx0 + 65 + i * 112
        h_init = (row["init_z"] / max_z) * (ly1 - ly0)
        h_att = (row["att_z"] / max_z) * (ly1 - ly0)

        lines.append(f'<rect x="{cx - 28}" y="{ly1 - h_init:.1f}" width="25" height="{h_init:.1f}" rx="4" fill="url(#gradBlue)"/>')
        lines.append(f'<text x="{cx - 15}" y="{ly1 - h_init - 6:.1f}" fill="#93c5fd" font-size="11" font-weight="600" text-anchor="middle">{row["init_z"]:.1f}</text>')

        lines.append(f'<rect x="{cx + 3}" y="{ly1 - h_att:.1f}" width="25" height="{h_att:.1f}" rx="4" fill="url(#gradGreen)"/>')
        lines.append(f'<text x="{cx + 15}" y="{ly1 - h_att - 6:.1f}" fill="#6ee7b7" font-size="11" font-weight="600" text-anchor="middle">{row["att_z"]:.1f}</text>')

        lines.append(f'<text x="{cx}" y="{ly1 + 22}" fill="#f1f5f9" font-size="12" font-weight="600" text-anchor="middle">{dname.capitalize()}</text>')
        scol = "#fbbf24" if dname == "academic" else "#34d399"
        lines.append(f'<text x="{cx}" y="{ly1 + 40}" fill="{scol}" font-size="11" text-anchor="middle">{row["surv"]*100:.1f}% survival</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "02_model_domain_comparison.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig3_survival_heatmap(merged: pd.DataFrame) -> str:
    """Figure 3: Heatmap of Survival Rate (%) across 18 Attack Configs x (3 Models + 4 Domains)."""
    W, H = 1100, 660
    lines = svg_header(W, H)

    lines.append('<text x="32" y="38" fill="#f8fafc" font-size="20" font-weight="700">Figure 3: Watermark Survival Rate (%) Heatmap — Attack × Model &amp; Domain</text>')
    lines.append('<text x="32" y="60" fill="#94a3b8" font-size="13">Percentage of documents retaining Z &gt;= 4.0 after each attack configuration (N = 600 documents × 18 configurations = 10,800 evaluations)</text>')

    order = (
        merged.groupby(["attack_name", "intensity"])["z_score_drop"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    m_piv = merged.pivot_table(index=["attack_name", "intensity"], columns="model_name", values="watermark_survived", aggfunc="mean").reindex(order)
    d_piv = merged.pivot_table(index=["attack_name", "intensity"], columns="domain", values="watermark_survived", aggfunc="mean").reindex(order)

    cols = [
        ("Qwen2.5-1.5B", m_piv["Qwen/Qwen2.5-1.5B-Instruct"]),
        ("SmolLM3-3B", m_piv["HuggingFaceTB/SmolLM3-3B"]),
        ("Llama-3.2-1B", m_piv["meta-llama/Llama-3.2-1B-Instruct"]),
        ("Creative", d_piv["creative"]),
        ("Conversational", d_piv["conversational"]),
        ("Instructional", d_piv["instructional"]),
        ("Academic", d_piv["academic"]),
    ]

    x0, y0 = 235, 108
    cw, ch = 115, 28

    lines.append(f'<text x="{x0 + 1.5*cw}" y="82" fill="#93c5fd" font-size="12" font-weight="700" text-anchor="middle">MODELS</text>')
    lines.append(f'<text x="{x0 + 5.1*cw}" y="82" fill="#c4b5fd" font-size="12" font-weight="700" text-anchor="middle">WRITING DOMAINS</text>')

    for c_idx, (cname, _) in enumerate(cols):
        cx = x0 + c_idx * cw + (15 if c_idx >= 3 else 0)
        lines.append(f'<text x="{cx + cw/2:.1f}" y="100" fill="#e2e8f0" font-size="11.5" font-weight="600" text-anchor="middle">{cname}</text>')

    for r_idx, (att, inten) in enumerate(order):
        cy = y0 + r_idx * ch
        label = f"{att} ({inten})"
        lines.append(f'<text x="{x0 - 12}" y="{cy + 18}" fill="#e2e8f0" font-size="11.5" text-anchor="end">{html.escape(label)}</text>')

        for c_idx, (_, series) in enumerate(cols):
            val = float(series.loc[(att, inten)]) * 100.0
            cx = x0 + c_idx * cw + (15 if c_idx >= 3 else 0)
            if val >= 95:
                bg_col, txt_col = "#065f46", "#ecfdf5"
            elif val >= 85:
                bg_col, txt_col = "#047857", "#f0fdf4"
            elif val >= 70:
                bg_col, txt_col = "#d97706", "#fffbeb"
            elif val >= 40:
                bg_col, txt_col = "#ea580c", "#fff7ed"
            else:
                bg_col, txt_col = "#991b1b", "#fef2f2"

            lines.append(f'<rect x="{cx + 2}" y="{cy + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" fill="{bg_col}"/>')
            lines.append(f'<text x="{cx + cw/2:.1f}" y="{cy + 18}" fill="{txt_col}" font-size="11" font-weight="600" text-anchor="middle">{val:.1f}%</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "03_survival_heatmap.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig4_feature_correlations(orig: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Figure 4: Diverging Bar Charts of Top Linguistic Features Driving Relative Z-Drop & Survival."""
    exclude_cols = {
        "doc_id", "file_name", "relative_path", "domain", "model_name",
        "initial_num_tokens_scored", "initial_num_green_tokens",
        "initial_green_fraction", "initial_z_score", "initial_p_value",
        "initial_detected", "initial_detection_threshold"
    }
    feature_cols = [c for c in orig.columns if c not in exclude_cols]

    doc_agg = merged.groupby("doc_id").agg(
        mean_rel_drop_pct=("relative_z_drop_pct", "mean"),
        survival_rate=("watermark_survived", "mean"),
    ).reset_index()
    doc_level = orig.merge(doc_agg, on="doc_id")

    rows = []
    for f in feature_cols:
        r_rel, _ = stats.pearsonr(doc_level[f], doc_level["mean_rel_drop_pct"])
        r_surv, _ = stats.pearsonr(doc_level[f], doc_level["survival_rate"])
        rows.append({"feature": f, "r_rel_drop": r_rel, "r_survival": r_surv})
    cdf = pd.DataFrame(rows)

    top_rel = cdf.reindex(cdf["r_rel_drop"].abs().sort_values(ascending=False).index).head(16).sort_values("r_rel_drop", ascending=True)
    top_surv = cdf.reindex(cdf["r_survival"].abs().sort_values(ascending=False).index).head(16).sort_values("r_survival", ascending=False)

    W, H = 1100, 600
    lines = svg_header(W, H)

    lines.append('<text x="32" y="38" fill="#f8fafc" font-size="20" font-weight="700">Figure 4: Which of the 42 Linguistic Features Govern Watermark Vulnerability?</text>')
    lines.append('<text x="32" y="60" fill="#94a3b8" font-size="13">Left: Pearson r with Relative Z-Score Drop (%) | Right: Pearson r with Post-Attack Watermark Survival Rate (N = 600 documents)</text>')

    lx_mid = 340
    scale = 380.0
    top_y, row_h = 105, 28
    lines.append('<text x="200" y="86" fill="#cbd5e1" font-size="12" font-weight="600">Top 16 Features Correlated with Relative Z-Score Drop (%)</text>')
    lines.append(f'<line x1="{lx_mid}" y1="{top_y - 6}" x2="{lx_mid}" y2="{top_y + 16*row_h}" stroke="#64748b" stroke-width="1.5"/>')

    for idx, (_, row) in enumerate(top_rel.iterrows()):
        cy = top_y + idx * row_h
        r = row["r_rel_drop"]
        bw = abs(r) * scale
        bx = lx_mid if r >= 0 else lx_mid - bw
        col = "#ef4444" if r >= 0 else "#10b981"
        lines.append(f'<text x="195" y="{cy + 17}" fill="#e2e8f0" font-size="11" text-anchor="end">{row["feature"]}</text>')
        lines.append(f'<rect x="{bx:.1f}" y="{cy + 4}" width="{bw:.1f}" height="18" rx="3" fill="{col}" fill-opacity="0.85"/>')
        tx = bx + bw + 5 if r >= 0 else bx - 5
        anchor = "start" if r >= 0 else "end"
        lines.append(f'<text x="{tx:.1f}" y="{cy + 17}" fill="#f1f5f9" font-size="10.5" font-weight="600" text-anchor="{anchor}">{r:+.3f}</text>')

    rx_mid = 850
    lines.append('<text x="700" y="86" fill="#cbd5e1" font-size="12" font-weight="600">Top 16 Features Correlated with Watermark Survival Rate</text>')
    lines.append(f'<line x1="{rx_mid}" y1="{top_y - 6}" x2="{rx_mid}" y2="{top_y + 16*row_h}" stroke="#64748b" stroke-width="1.5"/>')

    for idx, (_, row) in enumerate(top_surv.iterrows()):
        cy = top_y + idx * row_h
        r = row["r_survival"]
        bw = abs(r) * scale
        bx = rx_mid if r >= 0 else rx_mid - bw
        col = "#3b82f6" if r >= 0 else "#f97316"
        lines.append(f'<text x="705" y="{cy + 17}" fill="#e2e8f0" font-size="11" text-anchor="end">{row["feature"]}</text>')
        lines.append(f'<rect x="{bx:.1f}" y="{cy + 4}" width="{bw:.1f}" height="18" rx="3" fill="{col}" fill-opacity="0.85"/>')
        tx = bx + bw + 5 if r >= 0 else bx - 5
        anchor = "start" if r >= 0 else "end"
        lines.append(f'<text x="{tx:.1f}" y="{cy + 17}" fill="#f1f5f9" font-size="10.5" font-weight="600" text-anchor="{anchor}">{r:+.3f}</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "04_feature_correlations.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig5_attack_feature_matrix(merged: pd.DataFrame) -> str:
    """Figure 5: Attack Family x Key Linguistic Feature Correlation Matrix."""
    attacks = [
        ("humarin", "high", "Humarin (High)"),
        ("helm_typo", "high", "HELM Typo (High)"),
        ("copy_paste", "high", "Copy-Paste (High)"),
        ("synonym_wordnet", "high", "Synonym WordNet (High)"),
        ("helm_lowercase", "full", "HELM Lowercase (Full)"),
        ("helm_contraction", "full", "HELM Contraction (Full)"),
        ("helm_misspelling", "high", "HELM Misspelling (High)"),
    ]
    features = [
        "vocabulary_size",
        "bigram_entropy",
        "mattr",
        "smog_index",
        "gunning_fog_index",
        "adjective_ratio",
        "noun_ratio",
        "lexical_density",
        "stopword_ratio",
        "punctuation_density",
        "second_person_pronoun_ratio",
        "passive_voice_ratio",
    ]

    W, H = 1100, 520
    lines = svg_header(W, H)

    lines.append('<text x="32" y="38" fill="#f8fafc" font-size="20" font-weight="700">Figure 5: Attack-Specific Feature Vulnerability Matrix (Pearson r with Relative Z-Drop %)</text>')
    lines.append('<text x="32" y="60" fill="#94a3b8" font-size="13">Red (+r) = Feature increases vulnerability to that attack | Blue (-r) = Feature shields watermark from that attack</text>')

    x0, y0 = 240, 105
    cw, ch = 115, 29

    for c_idx, (_, _, alabel) in enumerate(attacks):
        cx = x0 + c_idx * cw
        lines.append(f'<text x="{cx + cw/2:.1f}" y="94" fill="#e2e8f0" font-size="10.5" font-weight="600" text-anchor="middle">{html.escape(alabel)}</text>')

    for r_idx, feat in enumerate(features):
        cy = y0 + r_idx * ch
        lines.append(f'<text x="{x0 - 12}" y="{cy + 19}" fill="#e2e8f0" font-size="11.5" text-anchor="end">{feat}</text>')

        for c_idx, (att, inten, _) in enumerate(attacks):
            sub = merged[(merged["attack_name"] == att) & (merged["intensity"] == inten)]
            r, _ = stats.pearsonr(sub[feat], sub["relative_z_drop_pct"])
            cx = x0 + c_idx * cw

            if r >= 0.30:
                bg_col = "#991b1b"
            elif r >= 0.15:
                bg_col = "#b45309"
            elif r <= -0.30:
                bg_col = "#1e40af"
            elif r <= -0.15:
                bg_col = "#0369a1"
            else:
                bg_col = "#1e293b"

            lines.append(f'<rect x="{cx + 2}" y="{cy + 2}" width="{cw - 4}" height="{ch - 4}" rx="4" fill="{bg_col}"/>')
            lines.append(f'<text x="{cx + cw/2:.1f}" y="{cy + 19}" fill="#f8fafc" font-size="11" font-weight="600" text-anchor="middle">{r:+.3f}</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "05_attack_feature_matrix.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def _draw_scatter_box(
    lines: list[str],
    x0: float, y0: float, x1: float, y1: float,
    x_vals: np.ndarray, y_vals: np.ndarray,
    colors: list[str],
    title: str, xlabel: str, ylabel: str,
    xmin: float = None, xmax: float = None,
    ymin: float = None, ymax: float = None,
    show_ols: bool = True,
    hline: float = None, hline_label: str = None,
    diagonal: bool = False,
    point_r: float = 3.2,
    point_opacity: float = 0.65,
):
    if xmin is None:
        xmin = float(np.nanmin(x_vals)) - 0.04 * (float(np.nanmax(x_vals)) - float(np.nanmin(x_vals)) or 1.0)
    if xmax is None:
        xmax = float(np.nanmax(x_vals)) + 0.04 * (float(np.nanmax(x_vals)) - float(np.nanmin(x_vals)) or 1.0)
    if ymin is None:
        ymin = float(np.nanmin(y_vals)) - 0.05 * (float(np.nanmax(y_vals)) - float(np.nanmin(y_vals)) or 1.0)
    if ymax is None:
        ymax = float(np.nanmax(y_vals)) + 0.06 * (float(np.nanmax(y_vals)) - float(np.nanmin(y_vals)) or 1.0)

    r_val, p_val = stats.pearsonr(x_vals, y_vals)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)

    lines.append(f'<text x="{x0}" y="{y0 - 10}" fill="#f1f5f9" font-size="12" font-weight="700">{html.escape(title)} <tspan fill="#38bdf8" font-weight="600">(r = {r_val:+.3f}, p = {p_val:.1e})</tspan></text>')
    lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="#1e293b" fill-opacity="0.45" stroke="#334155" rx="6"/>')

    # Grid ticks (4 ticks each axis)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xv = xmin + frac * (xmax - xmin)
        xx = x0 + frac * (x1 - x0)
        lines.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#334155" stroke-opacity="0.45" stroke-dasharray="2,3"/>')
        lbl = f"{xv:.2f}" if (xmax - xmin) < 5 else f"{xv:.0f}"
        lines.append(f'<text x="{xx:.1f}" y="{y1 + 14}" fill="#94a3b8" font-size="10" text-anchor="middle">{lbl}</text>')

        yv = ymin + frac * (ymax - ymin)
        yy = y1 - frac * (y1 - y0)
        lines.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#334155" stroke-opacity="0.45" stroke-dasharray="2,3"/>')
        ylbl = f"{yv:.1f}" if (ymax - ymin) < 25 else f"{yv:.0f}"
        lines.append(f'<text x="{x0 - 6}" y="{yy + 3.5:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{ylbl}</text>')

    lines.append(f'<text x="{(x0 + x1)/2:.1f}" y="{y1 + 29}" fill="#cbd5e1" font-size="11" font-weight="600" text-anchor="middle">{html.escape(xlabel)}</text>')
    lines.append(f'<text x="{x0 - 34}" y="{(y0 + y1)/2:.1f}" fill="#cbd5e1" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90 {x0 - 34} {(y0 + y1)/2:.1f})">{html.escape(ylabel)}</text>')

    if diagonal:
        d_min = max(xmin, ymin)
        d_max = min(xmax, ymax)
        dx0 = x0 + ((d_min - xmin) / (xmax - xmin)) * (x1 - x0)
        dy0 = y1 - ((d_min - ymin) / (ymax - ymin)) * (y1 - y0)
        dx1 = x0 + ((d_max - xmin) / (xmax - xmin)) * (x1 - x0)
        dy1 = y1 - ((d_max - ymin) / (ymax - ymin)) * (y1 - y0)
        lines.append(f'<line x1="{dx0:.1f}" y1="{dy0:.1f}" x2="{dx1:.1f}" y2="{dy1:.1f}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4,4"/>')

    if hline is not None and ymin <= hline <= ymax:
        hy = y1 - ((hline - ymin) / (ymax - ymin)) * (y1 - y0)
        lines.append(f'<line x1="{x0}" y1="{hy:.1f}" x2="{x1}" y2="{hy:.1f}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="5,4"/>')
        if hline_label:
            lines.append(f'<text x="{x1 - 6}" y="{hy - 5:.1f}" fill="#f87171" font-size="10" font-weight="600" text-anchor="end">{html.escape(hline_label)}</text>')

    for xv, yv, col in zip(x_vals, y_vals, colors):
        if xmin <= xv <= xmax and ymin <= yv <= ymax:
            px = x0 + ((xv - xmin) / (xmax - xmin)) * (x1 - x0)
            py = y1 - ((yv - ymin) / (ymax - ymin)) * (y1 - y0)
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{point_r}" fill="{col}" fill-opacity="{point_opacity}"/>')

    if show_ols:
        lx_start = xmin + 0.02 * (xmax - xmin)
        lx_end = xmax - 0.02 * (xmax - xmin)
        ly_start = np.clip(slope * lx_start + intercept, ymin, ymax)
        ly_end = np.clip(slope * lx_end + intercept, ymin, ymax)
        px0_l = x0 + ((lx_start - xmin) / (xmax - xmin)) * (x1 - x0)
        py0_l = y1 - ((ly_start - ymin) / (ymax - ymin)) * (y1 - y0)
        px1_l = x0 + ((lx_end - xmin) / (xmax - xmin)) * (x1 - x0)
        py1_l = y1 - ((ly_end - ymin) / (ymax - ymin)) * (y1 - y0)
        lines.append(f'<line x1="{px0_l:.1f}" y1="{py0_l:.1f}" x2="{px1_l:.1f}" y2="{py1_l:.1f}" stroke="#f8fafc" stroke-width="2" stroke-dasharray="6,3"/>')


def make_fig6_scatter_initial_vs_attacked(orig: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Figure 6: Scatter Plots of Initial Z-Score vs Post-Attack Z-Score (By Model & By Attack Family)."""
    W, H = 1100, 520
    lines = svg_header(W, H)

    lines.append('<text x="32" y="36" fill="#f8fafc" font-size="20" font-weight="700">Figure 6: Scatter Plot — Initial Z-Score vs. Post-Attack Z-Score (600 Documents)</text>')
    lines.append('<text x="32" y="56" fill="#94a3b8" font-size="13">Left: Document Mean Post-Attack Z colored by Model (N=600) | Right: High-Intensity Attacks colored by Attack Family (N=2,400 points)</text>')

    # Legend for Left Panel
    lines.append('<circle cx="70" cy="76" r="5" fill="#38bdf8"/><text x="80" y="80" fill="#e2e8f0" font-size="11">Qwen2.5-1.5B</text>')
    lines.append('<circle cx="185" cy="76" r="5" fill="#34d399"/><text x="195" y="80" fill="#e2e8f0" font-size="11">SmolLM3-3B</text>')
    lines.append('<circle cx="290" cy="76" r="5" fill="#fbbf24"/><text x="300" y="80" fill="#e2e8f0" font-size="11">Llama-3.2-1B</text>')

    doc_agg = merged.groupby("doc_id").agg(mean_att_z=("baseline_z_score", "mean")).reset_index()
    doc_df = orig.merge(doc_agg, on="doc_id")
    m_colors = {
        "Qwen/Qwen2.5-1.5B-Instruct": "#38bdf8",
        "HuggingFaceTB/SmolLM3-3B": "#34d399",
        "meta-llama/Llama-3.2-1B-Instruct": "#fbbf24",
    }
    cols_left = [m_colors.get(m, "#38bdf8") for m in doc_df["model_name"]]

    _draw_scatter_box(
        lines,
        x0=68, y0=108, x1=515, y1=455,
        x_vals=doc_df["initial_z_score"].to_numpy(),
        y_vals=doc_df["mean_att_z"].to_numpy(),
        colors=cols_left,
        title="A. Document Initial Z vs. Mean Attacked Z",
        xlabel="Initial Watermark Z-Score (Unattacked)",
        ylabel="Mean Post-Attack Z-Score",
        xmin=0, xmax=28, ymin=0, ymax=22,
        show_ols=True,
        hline=4.0, hline_label="Z = 4.0 Threshold",
        diagonal=True,
        point_r=3.5, point_opacity=0.72,
    )

    # Right Panel: 4 major attacks at high intensity (2,400 points) with per-attack OLS lines
    lines.append('<circle cx="605" cy="76" r="5" fill="#a855f7"/><text x="615" y="80" fill="#e2e8f0" font-size="11">Synonym (H)</text>')
    lines.append('<circle cx="705" cy="76" r="5" fill="#eab308"/><text x="715" y="80" fill="#e2e8f0" font-size="11">Copy-Paste (H)</text>')
    lines.append('<circle cx="820" cy="76" r="5" fill="#f97316"/><text x="830" y="80" fill="#e2e8f0" font-size="11">HELM Typo (H)</text>')
    lines.append('<circle cx="930" cy="76" r="5" fill="#ef4444"/><text x="940" y="80" fill="#e2e8f0" font-size="11">Humarin (H)</text>')

    sub_high = merged[
        (merged["intensity"] == "high")
        & (merged["attack_name"].isin(["synonym_wordnet", "copy_paste", "helm_typo", "humarin"]))
    ].copy()
    att_colors = {
        "synonym_wordnet": "#a855f7",
        "copy_paste": "#eab308",
        "helm_typo": "#f97316",
        "humarin": "#ef4444",
    }
    cols_right = [att_colors[a] for a in sub_high["attack_name"]]

    rx0, ry0, rx1, ry1 = 605, 108, 1060, 455
    _draw_scatter_box(
        lines,
        x0=rx0, y0=ry0, x1=rx1, y1=ry1,
        x_vals=sub_high["initial_z_score"].to_numpy(),
        y_vals=sub_high["baseline_z_score"].to_numpy(),
        colors=cols_right,
        title="B. Initial Z vs. Post-Attack Z by Attack Family (High)",
        xlabel="Initial Watermark Z-Score (Unattacked)",
        ylabel="Post-Attack Z-Score",
        xmin=0, xmax=28, ymin=-3, ymax=24,
        show_ols=False,
        hline=4.0, hline_label="Z = 4.0 Threshold",
        diagonal=True,
        point_r=2.4, point_opacity=0.45,
    )

    # Add per-attack OLS regression lines on Right Panel
    xmin, xmax, ymin, ymax = 0.0, 28.0, -3.0, 24.0
    for att_name, col in att_colors.items():
        s = sub_high[sub_high["attack_name"] == att_name]
        sl, ic = np.polyfit(s["initial_z_score"], s["baseline_z_score"], 1)
        x_s, x_e = 2.0, 26.5
        y_s, y_e = np.clip(sl * x_s + ic, ymin, ymax), np.clip(sl * x_e + ic, ymin, ymax)
        px0_l = rx0 + ((x_s - xmin) / (xmax - xmin)) * (rx1 - rx0)
        py0_l = ry1 - ((y_s - ymin) / (ymax - ymin)) * (ry1 - ry0)
        px1_l = rx0 + ((x_e - xmin) / (xmax - xmin)) * (rx1 - rx0)
        py1_l = ry1 - ((y_e - ymin) / (ymax - ymin)) * (ry1 - ry0)
        lines.append(f'<line x1="{px0_l:.1f}" y1="{py0_l:.1f}" x2="{px1_l:.1f}" y2="{py1_l:.1f}" stroke="{col}" stroke-width="2.5"/>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "06_scatter_initial_vs_attacked_z.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig7_scatter_grid_top_features(orig: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Figure 7: 2x3 Grid of Scatter Plots for Top 6 Linguistic Features vs Relative Z-Score Drop (%)."""
    W, H = 1100, 680
    lines = svg_header(W, H)

    lines.append('<text x="32" y="36" fill="#f8fafc" font-size="20" font-weight="700">Figure 7: Multi-Panel Scatter Grid — Top 6 Linguistic Features vs. Watermark Degradation</text>')
    lines.append('<text x="32" y="56" fill="#94a3b8" font-size="13">Each point is 1 document (N = 600), colored by Domain with OLS linear regression fit and Pearson correlation (r)</text>')

    # Domain legend
    dom_colors = {
        "academic": "#f87171",
        "conversational": "#38bdf8",
        "creative": "#c084fc",
        "instructional": "#34d399",
    }
    lx = 32
    for dname, dcol in dom_colors.items():
        lines.append(f'<circle cx="{lx + 6}" cy="75" r="5" fill="{dcol}"/><text x="{lx + 16}" y="79" fill="#e2e8f0" font-size="11.5" font-weight="600">{dname.capitalize()}</text>')
        lx += 130

    doc_agg = merged.groupby("doc_id").agg(
        mean_rel_drop_pct=("relative_z_drop_pct", "mean"),
        mean_z_drop=("z_score_drop", "mean"),
        survival_rate=("watermark_survived", "mean"),
    ).reset_index()
    doc_df = orig.merge(doc_agg, on="doc_id")
    syn_high = merged[(merged["attack_name"] == "synonym_wordnet") & (merged["intensity"] == "high")]
    doc_df = doc_df.merge(
        syn_high[["doc_id", "relative_z_drop_pct"]].rename(columns={"relative_z_drop_pct": "syn_rel_drop"}),
        on="doc_id",
    )

    colors = [dom_colors.get(d, "#38bdf8") for d in doc_df["domain"]]

    panels = [
        ("vocabulary_size", "mean_rel_drop_pct", "1. Vocabulary Size vs. Rel Z-Drop %", "Vocabulary Size (Unique Words)", "Mean Relative Z-Drop (%)"),
        ("smog_index", "mean_rel_drop_pct", "2. SMOG Readability vs. Rel Z-Drop %", "SMOG Grade Index", "Mean Relative Z-Drop (%)"),
        ("bigram_entropy", "mean_rel_drop_pct", "3. Bigram Entropy vs. Rel Z-Drop %", "Bigram Entropy (bits)", "Mean Relative Z-Drop (%)"),
        ("punctuation_density", "syn_rel_drop", "4. Punctuation vs. Synonym Z-Drop %", "Punctuation Density", "Synonym (High) Z-Drop (%)"),
        ("second_person_pronoun_ratio", "mean_rel_drop_pct", "5. 2nd-Person Pronouns vs. Rel Z-Drop %", "2nd-Person Pronoun Ratio", "Mean Relative Z-Drop (%)"),
        ("word_count", "initial_z_score", "6. Word Count vs. Initial Z-Score", "Document Word Count", "Initial Watermark Z-Score"),
    ]

    col_w, row_h = 310, 220
    x_starts = [62, 422, 775]
    y_starts = [110, 405]

    for idx, (fx, fy, title, xl, yl) in enumerate(panels):
        c = idx % 3
        r = idx // 3
        x0 = x_starts[c]
        y0 = y_starts[r]
        _draw_scatter_box(
            lines,
            x0=x0, y0=y0, x1=x0 + col_w, y1=y0 + row_h,
            x_vals=doc_df[fx].to_numpy(),
            y_vals=doc_df[fy].to_numpy(),
            colors=colors,
            title=title,
            xlabel=xl,
            ylabel=yl,
            show_ols=True,
            point_r=2.8,
            point_opacity=0.68,
        )

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "07_scatter_grid_top_features.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig8_semantic_cloud_and_trajectories(merged: pd.DataFrame) -> str:
    """Figure 8: Left: Point-Cloud Scatter of Cosine Similarity vs Attacked Z-Score | Right: Intensity Dose-Response Curves."""
    W, H = 1100, 520
    lines = svg_header(W, H)

    lines.append('<text x="32" y="36" fill="#f8fafc" font-size="20" font-weight="700">Figure 8: Semantic Similarity Point Cloud &amp; Intensity Dose-Response Curves</text>')
    lines.append('<text x="32" y="56" fill="#94a3b8" font-size="13">Left: Cosine Similarity vs. Post-Attack Z-Score (3,000 attacked samples) | Right: Z-Score Decay across Low → Medium → High Intensity</text>')

    # Left Panel: Point cloud of Semantic Similarity vs Post-Attack Z-score for 5 attacks
    sub = merged[
        (merged["intensity"].isin(["high", "full"]))
        & (merged["attack_name"].isin(["humarin", "helm_typo", "copy_paste", "synonym_wordnet", "helm_lowercase"]))
    ].copy()
    sub = sub[sub["semantic_similarity_cosine"] >= 0.60].copy()

    att_cols = {
        "humarin": "#ef4444",
        "helm_typo": "#f97316",
        "copy_paste": "#eab308",
        "synonym_wordnet": "#a855f7",
        "helm_lowercase": "#06b6d4",
    }
    cols = [att_cols[a] for a in sub["attack_name"]]

    lx = 68
    for aname, acol in [("Humarin", "#ef4444"), ("Typo", "#f97316"), ("CopyPaste", "#eab308"), ("Synonym", "#a855f7"), ("Lowercase", "#06b6d4")]:
        lines.append(f'<circle cx="{lx}" cy="76" r="5" fill="{acol}"/><text x="{lx + 9}" y="80" fill="#e2e8f0" font-size="11">{aname}</text>')
        lx += 90

    _draw_scatter_box(
        lines,
        x0=68, y0=108, x1=520, y1=455,
        x_vals=sub["semantic_similarity_cosine"].to_numpy(),
        y_vals=sub["baseline_z_score"].to_numpy(),
        colors=cols,
        title="A. Semantic Similarity vs. Post-Attack Z-Score",
        xlabel="Cosine Semantic Similarity (Watermarked vs. Attacked)",
        ylabel="Post-Attack KGW Z-Score",
        xmin=0.60, xmax=1.01, ymin=-3, ymax=24,
        show_ols=True,
        hline=4.0, hline_label="Survival Threshold (Z = 4.0)",
        point_r=2.5, point_opacity=0.50,
    )

    # Right Panel: Dose-response trajectories (Baseline -> Low -> Medium -> High) with IQR error bars
    rx0, ry0, rx1, ry1 = 605, 108, 1060, 455
    lines.append(f'<text x="{rx0}" y="{ry0 - 10}" fill="#f1f5f9" font-size="12" font-weight="700">B. Dose-Response Trajectories (Mean Z ± IQR across Intensities)</text>')
    lines.append(f'<rect x="{rx0}" y="{ry0}" width="{rx1 - rx0}" height="{ry1 - ry0}" fill="#1e293b" fill-opacity="0.45" stroke="#334155" rx="6"/>')

    ymin, ymax = 0.0, 20.0
    for yv in [0, 4, 8, 12, 16, 20]:
        yy = ry1 - ((yv - ymin) / (ymax - ymin)) * (ry1 - ry0)
        col_line = "#ef4444" if yv == 4 else "#334155"
        dash = "5,4" if yv == 4 else "2,3"
        lines.append(f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx1}" y2="{yy:.1f}" stroke="{col_line}" stroke-dasharray="{dash}"/>')
        lines.append(f'<text x="{rx0 - 6}" y="{yy + 4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{yv}</text>')

    stages = ["Unattacked", "Low (0.10)", "Medium (0.30)", "High (0.50)"]
    xs = [rx0 + 45 + i * 120 for i in range(4)]
    for i, st in enumerate(stages):
        lines.append(f'<line x1="{xs[i]}" y1="{ry0}" x2="{xs[i]}" y2="{ry1}" stroke="#334155" stroke-opacity="0.4"/>')
        lines.append(f'<text x="{xs[i]}" y="{ry1 + 18}" fill="#cbd5e1" font-size="11" font-weight="600" text-anchor="middle">{st}</text>')

    init_mean = merged["initial_z_score"].mean()
    for offset, (att_name, col, label) in enumerate([
        ("synonym_wordnet", "#a855f7", "synonym_wordnet"),
        ("copy_paste", "#eab308", "copy_paste"),
        ("helm_typo", "#f97316", "helm_typo"),
        ("humarin", "#ef4444", "humarin"),
    ]):
        pts = [(xs[0], ry1 - (init_mean / ymax) * (ry1 - ry0))]
        for s_idx, inten in enumerate(["low", "medium", "high"], start=1):
            sub_i = merged[(merged["attack_name"] == att_name) & (merged["intensity"] == inten)]["baseline_z_score"]
            m_val = sub_i.mean()
            q25, q75 = np.percentile(sub_i, [25, 75])
            px = xs[s_idx] + (offset - 1.5) * 3.5
            py = ry1 - (np.clip(m_val, ymin, ymax) / ymax) * (ry1 - ry0)
            py25 = ry1 - (np.clip(q25, ymin, ymax) / ymax) * (ry1 - ry0)
            py75 = ry1 - (np.clip(q75, ymin, ymax) / ymax) * (ry1 - ry0)
            lines.append(f'<line x1="{px:.1f}" y1="{py25:.1f}" x2="{px:.1f}" y2="{py75:.1f}" stroke="{col}" stroke-width="1.8" stroke-opacity="0.7"/>')
            lines.append(f'<line x1="{px - 4:.1f}" y1="{py25:.1f}" x2="{px + 4:.1f}" y2="{py25:.1f}" stroke="{col}" stroke-width="1.8"/>')
            lines.append(f'<line x1="{px - 4:.1f}" y1="{py75:.1f}" x2="{px + 4:.1f}" y2="{py75:.1f}" stroke="{col}" stroke-width="1.8"/>')
            pts.append((px, py))

        path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        lines.append(f'<path d="{path_d}" fill="none" stroke="{col}" stroke-width="2.8"/>')
        for x, y in pts:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="#0f172a" stroke-width="1.2"/>')
        lines.append(f'<text x="{pts[-1][0] + 8:.1f}" y="{pts[-1][1] + 4:.1f}" fill="{col}" font-size="10.5" font-weight="700">{label}</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "08_scatter_semantic_sim_vs_z_drop_cloud.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


def make_fig9_box_and_jitter_distributions(orig: pd.DataFrame, merged: pd.DataFrame) -> str:
    """Figure 9: Box-and-Whisker + Jittered Strip Scatter Plots of Z-Score Distributions by Model & Domain."""
    W, H = 1100, 500
    lines = svg_header(W, H)

    lines.append('<text x="32" y="36" fill="#f8fafc" font-size="20" font-weight="700">Figure 9: Distributional Boxplots &amp; Jittered Scatter — By Model and Writing Domain</text>')
    lines.append('<text x="32" y="56" fill="#94a3b8" font-size="13">Boxes show Median &amp; IQR (25th–75th percentile); whiskers show 10th–90th percentile; dots show individual documents (N = 600)</text>')

    rng = np.random.default_rng(42)

    # Left Panel: Initial Z-Score distribution across Models
    lx0, ly0, lx1, ly1 = 68, 100, 520, 440
    lines.append(f'<text x="{lx0}" y="{ly0 - 10}" fill="#f1f5f9" font-size="12" font-weight="700">A. Initial Watermark Z-Score Distribution by Model (N = 200 each)</text>')
    lines.append(f'<rect x="{lx0}" y="{ly0}" width="{lx1 - lx0}" height="{ly1 - ly0}" fill="#1e293b" fill-opacity="0.45" stroke="#334155" rx="6"/>')

    ymin, ymax = 0.0, 28.0
    for yv in [0, 4, 8, 12, 16, 20, 24, 28]:
        yy = ly1 - (yv / ymax) * (ly1 - ly0)
        col_l = "#ef4444" if yv == 4 else "#334155"
        lines.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx1}" y2="{yy:.1f}" stroke="{col_l}" stroke-opacity="0.55" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{lx0 - 6}" y="{yy + 4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{yv}</text>')

    model_specs = [
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B", "#38bdf8"),
        ("HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B", "#34d399"),
        ("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B", "#fbbf24"),
    ]
    for idx, (m_id, m_label, col) in enumerate(model_specs):
        vals = orig[orig["model_name"] == m_id]["initial_z_score"].to_numpy()
        cx = lx0 + 80 + idx * 145
        q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90])

        # Jittered strip points
        jit = rng.uniform(-22, 22, size=len(vals))
        for v, jx in zip(vals, jit):
            py = ly1 - (np.clip(v, ymin, ymax) / ymax) * (ly1 - ly0)
            lines.append(f'<circle cx="{cx + jx:.1f}" cy="{py:.1f}" r="2.6" fill="{col}" fill-opacity="0.45"/>')

        y10 = ly1 - (q10 / ymax) * (ly1 - ly0)
        y25 = ly1 - (q25 / ymax) * (ly1 - ly0)
        y50 = ly1 - (q50 / ymax) * (ly1 - ly0)
        y75 = ly1 - (q75 / ymax) * (ly1 - ly0)
        y90 = ly1 - (q90 / ymax) * (ly1 - ly0)

        lines.append(f'<line x1="{cx}" y1="{y10:.1f}" x2="{cx}" y2="{y90:.1f}" stroke="#f8fafc" stroke-width="1.5"/>')
        lines.append(f'<line x1="{cx - 14}" y1="{y10:.1f}" x2="{cx + 14}" y2="{y10:.1f}" stroke="#f8fafc" stroke-width="1.5"/>')
        lines.append(f'<line x1="{cx - 14}" y1="{y90:.1f}" x2="{cx + 14}" y2="{y90:.1f}" stroke="#f8fafc" stroke-width="1.5"/>')
        lines.append(f'<rect x="{cx - 28}" y="{y75:.1f}" width="56" height="{max(2.0, y25 - y75):.1f}" fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="2" rx="3"/>')
        lines.append(f'<line x1="{cx - 28}" y1="{y50:.1f}" x2="{cx + 28}" y2="{y50:.1f}" stroke="#ffffff" stroke-width="2.5"/>')
        lines.append(f'<text x="{cx}" y="{ly1 + 20}" fill="#f1f5f9" font-size="11.5" font-weight="600" text-anchor="middle">{m_label}</text>')
        lines.append(f'<text x="{cx}" y="{ly1 + 35}" fill="{col}" font-size="10.5" text-anchor="middle">med={q50:.1f} (IQR {q25:.1f}–{q75:.1f})</text>')

    # Right Panel: Mean Post-Attack Z-Score per Document across Writing Domains
    doc_agg = merged.groupby("doc_id").agg(mean_att_z=("baseline_z_score", "mean")).reset_index()
    doc_df = orig.merge(doc_agg, on="doc_id")

    rx0, ry0, rx1, ry1 = 600, 100, 1065, 440
    lines.append(f'<text x="{rx0}" y="{ry0 - 10}" fill="#f1f5f9" font-size="12" font-weight="700">B. Document Mean Post-Attack Z-Score by Writing Domain (N = 150 each)</text>')
    lines.append(f'<rect x="{rx0}" y="{ry0}" width="{rx1 - rx0}" height="{ry1 - ry0}" fill="#1e293b" fill-opacity="0.45" stroke="#334155" rx="6"/>')

    ymin_r, ymax_r = 0.0, 20.0
    for yv in [0, 4, 8, 12, 16, 20]:
        yy = ry1 - (yv / ymax_r) * (ry1 - ry0)
        col_l = "#ef4444" if yv == 4 else "#334155"
        lines.append(f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx1}" y2="{yy:.1f}" stroke="{col_l}" stroke-opacity="0.55" stroke-dasharray="3,3"/>')
        lines.append(f'<text x="{rx0 - 6}" y="{yy + 4:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{yv}</text>')

    dom_specs = [
        ("creative", "Creative", "#c084fc"),
        ("conversational", "Conversational", "#38bdf8"),
        ("instructional", "Instructional", "#34d399"),
        ("academic", "Academic", "#f87171"),
    ]
    for idx, (d_id, d_label, col) in enumerate(dom_specs):
        vals = doc_df[doc_df["domain"] == d_id]["mean_att_z"].to_numpy()
        cx = rx0 + 62 + idx * 112
        q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90])

        jit = rng.uniform(-18, 18, size=len(vals))
        for v, jx in zip(vals, jit):
            py = ry1 - (np.clip(v, ymin_r, ymax_r) / ymax_r) * (ry1 - ry0)
            lines.append(f'<circle cx="{cx + jx:.1f}" cy="{py:.1f}" r="2.6" fill="{col}" fill-opacity="0.48"/>')

        y10 = ry1 - (q10 / ymax_r) * (ry1 - ry0)
        y25 = ry1 - (q25 / ymax_r) * (ry1 - ry0)
        y50 = ry1 - (q50 / ymax_r) * (ry1 - ry0)
        y75 = ry1 - (q75 / ymax_r) * (ry1 - ry0)
        y90 = ry1 - (q90 / ymax_r) * (ry1 - ry0)

        lines.append(f'<line x1="{cx}" y1="{y10:.1f}" x2="{cx}" y2="{y90:.1f}" stroke="#f8fafc" stroke-width="1.5"/>')
        lines.append(f'<line x1="{cx - 12}" y1="{y10:.1f}" x2="{cx + 12}" y2="{y10:.1f}" stroke="#f8fafc" stroke-width="1.5"/>')
        lines.append(f'<line x1="{cx - 12}" y1="{y90:.1f}" x2="{cx + 12}" y2="{y90:.1f}" stroke="#f8fafc" stroke-width="1.5"/>')
        lines.append(f'<rect x="{cx - 24}" y="{y75:.1f}" width="48" height="{max(2.0, y25 - y75):.1f}" fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="2" rx="3"/>')
        lines.append(f'<line x1="{cx - 24}" y1="{y50:.1f}" x2="{cx + 24}" y2="{y50:.1f}" stroke="#ffffff" stroke-width="2.5"/>')
        lines.append(f'<text x="{cx}" y="{ry1 + 20}" fill="#f1f5f9" font-size="11.5" font-weight="600" text-anchor="middle">{d_label}</text>')
        lines.append(f'<text x="{cx}" y="{ry1 + 35}" fill="{col}" font-size="10" text-anchor="middle">med={q50:.1f}</text>')

    lines.append('</svg>')
    svg_str = "\n".join(lines)
    (PLOTS_DIR / "09_boxplots_distributions_by_model_domain.svg").write_text(svg_str, encoding="utf-8")
    return svg_str


svg1 = make_fig1_attack_ranking(merged_df)
svg2 = make_fig2_model_domain_comparison(orig_df, merged_df)
svg3 = make_fig3_survival_heatmap(merged_df)
svg4 = make_fig4_feature_correlations(orig_df, merged_df)
svg5 = make_fig5_attack_feature_matrix(merged_df)
svg6 = make_fig6_scatter_initial_vs_attacked(orig_df, merged_df)
svg7 = make_fig7_scatter_grid_top_features(orig_df, merged_df)
svg8 = make_fig8_semantic_cloud_and_trajectories(merged_df)
svg9 = make_fig9_box_and_jitter_distributions(orig_df, merged_df)

print("Generated all 9 SVG charts in:", PLOTS_DIR)

cells = []


def add_markdown(md_text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in md_text.strip().split("\n")]
    })


exec_ns = {}


def add_code(code_text, exec_count, svg_list=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code_text, exec_ns)
    out_str = buf.getvalue()
    outputs = []
    if out_str:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in out_str.rstrip("\n").split("\n")]
        })
    if svg_list:
        for svg_content in svg_list:
            outputs.append({
                "data": {
                    "image/svg+xml": [line + "\n" for line in svg_content.splitlines()],
                    "text/html": [f'<div style="margin: 12px 0;">{svg_content}</div>\n'],
                    "text/plain": ["<IPython.core.display.SVG object>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_text.strip().split("\n")]
    })


add_markdown("""# Exploratory Data Analysis (EDA): KGW Watermark Robustness Prediction
This notebook analyzes the merged dataset (`merged_robustness_dataset.csv`, **10,800 attacked samples** across **600 original documents**) combined from `original_features_combined.csv` and `attack_settings_combined.csv`.

### Key Research Questions Answered:
1. **Baseline Embeddability & Distributions**: How does initial KGW watermark strength (`initial_z_score`) distribute across LLMs (`Qwen2.5-1.5B`, `SmolLM3-3B`, `Llama-3.2-1B`) and text domains (`academic`, `conversational`, `creative`, `instructional`)?
2. **Attack Effectiveness, Dose-Response & Semantic Trade-off**: Which attacks cause the largest absolute (`z_score_drop`) and relative (`relative_z_drop_pct`) drop in z-score, and how do individual documents scatter across semantic similarity vs. post-attack z-score?
3. **Model & Domain Robustness**: Which LLM and writing domain produce watermarks that best survive attacks?
4. **Linguistic Feature Drivers (Scatter & Correlation Analysis)**: Which of the 42 linguistic, syntactic, readability, and entropy features most strongly govern watermark degradation overall and per attack family?""")

add_code("""from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
try:
    from IPython.display import SVG, display
except ImportError:
    SVG = lambda filename=None, **kw: filename
    display = lambda *a, **kw: None

DATA_DIR = Path(".") if Path("original_features_combined.csv").exists() else Path("src")
PLOTS_DIR = DATA_DIR / "plots"
orig_df = pd.read_csv(DATA_DIR / "original_features_combined.csv")
merged_df = pd.read_csv(DATA_DIR / "merged_robustness_dataset.csv", low_memory=False)

print(f"Original features dataset shape : {orig_df.shape}")
print(f"Merged robustness dataset shape : {merged_df.shape}")
print(f"Unique documents (doc_id)       : {merged_df['doc_id'].nunique()}")
print(f"Attack configurations per doc   : {len(merged_df) // merged_df['doc_id'].nunique()}")
print(f"Models                          : {list(orig_df['model_name'].unique())}")
print(f"Domains                         : {list(orig_df['domain'].unique())}")""", 1)

add_markdown("""## 1. Initial Watermark Strength, Boxplots & Post-Attack Comparison by Model and Domain
We inspect both mean summary bars and full **box-and-whisker + jittered scatter distributions** across all 600 documents.""")

add_code("""model_init = orig_df.groupby("model_name").agg(
    docs=("doc_id", "count"),
    mean_initial_z=("initial_z_score", "mean"),
    std_initial_z=("initial_z_score", "std"),
    median_initial_z=("initial_z_score", "median"),
    mean_green_frac=("initial_green_fraction", "mean"),
    initial_detect_rate=("initial_detected", "mean"),
    mean_word_count=("word_count", "mean"),
    mean_vocab_size=("vocabulary_size", "mean"),
    mean_bigram_entropy=("bigram_entropy", "mean"),
).round(4)
print("=== INITIAL WATERMARK STATS BY MODEL ===")
print(model_init.to_string())

domain_init = orig_df.groupby("domain").agg(
    docs=("doc_id", "count"),
    mean_initial_z=("initial_z_score", "mean"),
    std_initial_z=("initial_z_score", "std"),
    mean_green_frac=("initial_green_fraction", "mean"),
    initial_detect_rate=("initial_detected", "mean"),
    mean_word_count=("word_count", "mean"),
    mean_mattr=("mattr", "mean"),
    mean_bigram_entropy=("bigram_entropy", "mean"),
).round(4)
print("\\n=== INITIAL WATERMARK STATS BY DOMAIN ===")
print(domain_init.to_string())
display(SVG(filename=str(PLOTS_DIR / "02_model_domain_comparison.svg")))
display(SVG(filename=str(PLOTS_DIR / "09_boxplots_distributions_by_model_domain.svg")))""", 2, svg_list=[svg2, svg9])

add_markdown("""## 2. Scatter Analysis: Initial Z-Score vs. Post-Attack Z-Score & Semantic Similarity Cloud
Here we plot:
- **Figure 1**: Attack Effectiveness Ranking & Pareto Frontier.
- **Figure 6**: Scatter plots of `initial_z_score` vs. `post_attack_z_score` colored by Model (left) and by Attack Family with OLS regression slopes (right).
- **Figure 8**: Point-cloud scatter of `semantic_similarity_cosine` vs. `baseline_z_score` (left) and intensity dose-response decay curves (`Low -> Medium -> High`) with IQR error bars (right).""")

add_code("""attack_rank = merged_df.groupby(["attack_name", "intensity"]).agg(
    n=("doc_id", "count"),
    mean_initial_z=("initial_z_score", "mean"),
    mean_attacked_z=("baseline_z_score", "mean"),
    mean_z_drop=("z_score_drop", "mean"),
    std_z_drop=("z_score_drop", "std"),
    mean_rel_drop_pct=("relative_z_drop_pct", "mean"),
    survival_rate=("watermark_survived", "mean"),
    removal_rate=("watermark_removed", "mean"),
    semantic_sim=("semantic_similarity_cosine", "mean"),
).round(4).sort_values("mean_z_drop", ascending=False)

print("=== ATTACK EFFECTIVENESS RANKING (SORTED BY Z-SCORE DROP) ===")
print(attack_rank.to_string())
display(SVG(filename=str(PLOTS_DIR / "01_attack_pareto_ranking.svg")))
display(SVG(filename=str(PLOTS_DIR / "06_scatter_initial_vs_attacked_z.svg")))
display(SVG(filename=str(PLOTS_DIR / "08_scatter_semantic_sim_vs_z_drop_cloud.svg")))""", 3, svg_list=[svg1, svg6, svg8])

add_markdown("""## 3. Model & Domain Survival Rate Heatmap Across All 18 Attack Configurations
We compare how each model (`Qwen2.5-1.5B`, `SmolLM3-3B`, `Llama-3.2-1B`) and each writing domain (`Creative`, `Conversational`, `Instructional`, `Academic`) survives each of the 18 attack configurations.""")

add_code("""model_perf = merged_df.groupby("model_name").agg(
    total_attacked=("doc_id", "count"),
    mean_initial_z=("initial_z_score", "mean"),
    mean_attacked_z=("baseline_z_score", "mean"),
    mean_z_drop=("z_score_drop", "mean"),
    mean_rel_drop_pct=("relative_z_drop_pct", "mean"),
    overall_survival_rate=("watermark_survived", "mean"),
    non_paraphrase_survival=("watermark_survived", lambda s: s[merged_df.loc[s.index, "attack_name"] != "humarin"].mean()),
    paraphrase_survival=("watermark_survived", lambda s: s[merged_df.loc[s.index, "attack_name"] == "humarin"].mean()),
    mean_semantic_sim=("semantic_similarity_cosine", "mean"),
).round(4)

print("=== OVERALL MODEL PERFORMANCE UNDER ATTACK ===")
print(model_perf.to_string())

domain_perf = merged_df.groupby("domain").agg(
    total_attacked=("doc_id", "count"),
    mean_initial_z=("initial_z_score", "mean"),
    mean_attacked_z=("baseline_z_score", "mean"),
    mean_z_drop=("z_score_drop", "mean"),
    mean_rel_drop_pct=("relative_z_drop_pct", "mean"),
    overall_survival_rate=("watermark_survived", "mean"),
    non_paraphrase_survival=("watermark_survived", lambda s: s[merged_df.loc[s.index, "attack_name"] != "humarin"].mean()),
    mean_semantic_sim=("semantic_similarity_cosine", "mean"),
).round(4)

print("\\n=== DOMAIN ROBUSTNESS SUMMARY ===")
print(domain_perf.to_string())
display(SVG(filename=str(PLOTS_DIR / "03_survival_heatmap.svg")))""", 4, svg_list=[svg3])

add_markdown("""## 4. Feature Correlations & 2×3 Scatter Plot Grid of Top Linguistic Drivers
We analyze correlations across all 42 linguistic features and plot a **6-panel scatter grid (`07_scatter_grid_top_features.svg`)** with OLS linear trend lines and domain color-coding.""")

add_code("""exclude_cols = {
    "doc_id", "file_name", "relative_path", "domain", "model_name",
    "initial_num_tokens_scored", "initial_num_green_tokens",
    "initial_green_fraction", "initial_z_score", "initial_p_value",
    "initial_detected", "initial_detection_threshold"
}
feature_cols = [c for c in orig_df.columns if c not in exclude_cols]

doc_agg = merged_df.groupby("doc_id").agg(
    mean_z_drop=("z_score_drop", "mean"),
    mean_rel_drop_pct=("relative_z_drop_pct", "mean"),
    mean_attacked_z=("baseline_z_score", "mean"),
    survival_rate=("watermark_survived", "mean"),
).reset_index()

doc_level = orig_df.merge(doc_agg, on="doc_id")

rows = []
for f in feature_cols + ["initial_z_score", "initial_green_fraction", "initial_num_tokens_scored"]:
    r_init, _ = stats.pearsonr(doc_level[f], doc_level["initial_z_score"])
    r_abs, p_abs = stats.pearsonr(doc_level[f], doc_level["mean_z_drop"])
    r_rel, p_rel = stats.pearsonr(doc_level[f], doc_level["mean_rel_drop_pct"])
    r_surv, p_surv = stats.pearsonr(doc_level[f], doc_level["survival_rate"])
    rho_surv, _ = stats.spearmanr(doc_level[f], doc_level["survival_rate"])
    rows.append({
        "feature": f,
        "r_initial_z": r_init,
        "r_abs_z_drop": r_abs,
        "r_rel_z_drop_pct": r_rel,
        "r_survival_rate": r_surv,
        "spearman_survival": rho_surv,
    })

corr_table = pd.DataFrame(rows)

print("=== TOP 15 FEATURES BY RELATIVE Z-SCORE DROP (%) CORRELATION ===")
print(corr_table.reindex(corr_table["r_rel_z_drop_pct"].abs().sort_values(ascending=False).index).head(15).round(4).to_string(index=False))

print("\\n=== TOP 15 FEATURES BY POST-ATTACK SURVIVAL RATE CORRELATION ===")
print(corr_table.reindex(corr_table["r_survival_rate"].abs().sort_values(ascending=False).index).head(15).round(4).to_string(index=False))
display(SVG(filename=str(PLOTS_DIR / "04_feature_correlations.svg")))
display(SVG(filename=str(PLOTS_DIR / "07_scatter_grid_top_features.svg")))""", 5, svg_list=[svg4, svg7])

add_markdown("""## 5. Attack-Specific Feature Vulnerability Matrix
Different attack mechanisms exploit different linguistic properties. Below, we compute the correlation matrix between linguistic features and relative z-score drop (`relative_z_drop_pct`) for each major attack family at high/full intensity.""")

add_code("""for att in ["humarin", "helm_typo", "copy_paste", "synonym_wordnet", "helm_lowercase", "helm_contraction", "helm_misspelling"]:
    sub = merged_df[merged_df["attack_name"] == att]
    if "high" in sub["intensity"].values:
        sub = sub[sub["intensity"] == "high"]
    att_corrs = []
    for f in feature_cols:
        r_rel, p_val = stats.pearsonr(sub[f], sub["relative_z_drop_pct"])
        r_abs, _ = stats.pearsonr(sub[f], sub["z_score_drop"])
        att_corrs.append((f, r_rel, r_abs, p_val))
    att_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"=== {att.upper()} ({sub['intensity'].iloc[0]}) — Top 5 Linguistic Drivers of Relative Z-Drop ===")
    for f, r_rel, r_abs, p_val in att_corrs[:5]:
        print(f"  {f:30s} | r_rel_drop = {r_rel:+.4f} (p={p_val:.2e}) | r_abs_drop = {r_abs:+.4f}")
    print()
display(SVG(filename=str(PLOTS_DIR / "05_attack_feature_matrix.svg")))""", 6, svg_list=[svg5])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_nb = SRC_DIR / "eda_robustness_analysis.ipynb"
out_nb.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Updated executed notebook with all 9 embedded SVG charts at:", out_nb)
