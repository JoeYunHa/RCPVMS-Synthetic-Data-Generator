"""RCP VMS Normal Data Cluster Analysis — PDF Report Generator

Output: data/analysis/normal_cluster/normal_cluster_analysis_report.pdf
"""

from pathlib import Path
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)

# ── Paths ─────────────────────────────────────────────────────────────────────
IMG_DIR = Path("data/analysis/normal_cluster")
OUT_PDF = IMG_DIR / "normal_cluster_analysis_report.pdf"
W, H    = A4

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY  = colors.HexColor("#1A237E")
BLUE  = colors.HexColor("#1565C0")
TEAL  = colors.HexColor("#00695C")
RED   = colors.HexColor("#B71C1C")
LGRAY = colors.HexColor("#F5F5F5")
MGRAY = colors.HexColor("#BDBDBD")
DGRAY = colors.HexColor("#424242")

CONTENT_W = W - 4 * cm

# ── Paragraph styles ──────────────────────────────────────────────────────────
title_style = ParagraphStyle("ReportTitle",
    fontSize=22, fontName="Helvetica-Bold",
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)

subtitle_style = ParagraphStyle("ReportSubtitle",
    fontSize=12, fontName="Helvetica",
    textColor=DGRAY, alignment=TA_CENTER, spaceAfter=4)

meta_style = ParagraphStyle("Meta",
    fontSize=10, fontName="Helvetica",
    textColor=colors.grey, alignment=TA_CENTER, spaceAfter=2)

h1_style = ParagraphStyle("H1",
    fontSize=15, fontName="Helvetica-Bold",
    textColor=NAVY, spaceBefore=18, spaceAfter=8, leading=18)

h2_style = ParagraphStyle("H2",
    fontSize=12, fontName="Helvetica-Bold",
    textColor=BLUE, spaceBefore=12, spaceAfter=6)

h3_style = ParagraphStyle("H3",
    fontSize=10, fontName="Helvetica-BoldOblique",
    textColor=TEAL, spaceBefore=8, spaceAfter=4)

body_style = ParagraphStyle("Body",
    fontSize=9.5, fontName="Helvetica",
    textColor=DGRAY, leading=15, spaceAfter=6,
    alignment=TA_JUSTIFY)

caption_style = ParagraphStyle("Caption",
    fontSize=8.5, fontName="Helvetica-Oblique",
    textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)

note_style = ParagraphStyle("Note",
    fontSize=8.5, fontName="Helvetica-Oblique",
    textColor=colors.grey, spaceAfter=6)


# ── Helper functions ──────────────────────────────────────────────────────────
def fig(name: str, width=None, height=None) -> Image:
    w = width or CONTENT_W
    return Image(str(IMG_DIR / name), width=w, height=height or w * 0.55)


def section_line():
    return HRFlowable(width="100%", thickness=1.2,
                      color=NAVY, spaceAfter=6, spaceBefore=2)


def kv_table(rows, col_widths=None):
    cw = col_widths or [4.5 * cm, CONTENT_W - 4.5 * cm]
    data = [[Paragraph(f"<b>{k}</b>", body_style),
             Paragraph(v, body_style)] for k, v in rows]
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), LGRAY),
        ("GRID",          (0, 0), (-1, -1), 0.4, MGRAY),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def data_table(header, rows, col_widths=None):
    th_style = ParagraphStyle("TH", fontSize=9, fontName="Helvetica-Bold",
                               textColor=colors.white, alignment=TA_CENTER)
    td_style = ParagraphStyle("TD", fontSize=9, fontName="Helvetica",
                               alignment=TA_CENTER)
    data = [[Paragraph(h, th_style) for h in header]]
    for row in rows:
        data.append([Paragraph(str(c), td_style) for c in row])
    n = len(header)
    cw = col_widths or [CONTENT_W / n] * n
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, MGRAY),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def conclusion_box(text):
    cell = Paragraph(text, ParagraphStyle("concl",
        fontSize=9.5, fontName="Helvetica-Bold",
        textColor=NAVY, leading=15))
    t = Table([[cell]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#E8EAF6")),
        ("BOX",           (0, 0), (-1, -1), 1.5, NAVY),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def warn_box(text):
    cell = Paragraph(text, ParagraphStyle("warnb",
        fontSize=9.5, fontName="Helvetica-Bold",
        textColor=RED, leading=15))
    t = Table([[cell]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFEBEE")),
        ("BOX",           (0, 0), (-1, -1), 1.5, RED),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ── Page number callback ──────────────────────────────────────────────────────
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"- {doc.page} -")
    canvas.drawString(2*cm, 1.2*cm,
                      "RCP VMS Normal Data Cluster Analysis Report")
    canvas.restoreState()


# ── Story ─────────────────────────────────────────────────────────────────────
def build_story():
    s = []

    # ── Cover page ────────────────────────────────────────────────────────────
    s.append(Spacer(1, 3*cm))
    s.append(Paragraph("RCP VMS Normal Data", title_style))
    s.append(Paragraph("Cluster Structure Analysis Report", title_style))
    s.append(Spacer(1, 0.4*cm))
    s.append(HRFlowable(width="60%", thickness=2, color=NAVY,
                         hAlign="CENTER", spaceAfter=12))
    s.append(Paragraph(
        "Pre-validation of Deep SVDD assumption: "
        "single-cluster vs. multi-modal structure in normal operating data",
        subtitle_style))
    s.append(Spacer(1, 1*cm))
    s.append(Paragraph(f"Date: {date.today().strftime('%B %d, %Y')}", meta_style))
    s.append(Paragraph(
        "Dataset: 110 measured normal BIN files  (1200 rpm: 82  /  3600 rpm: 28)",
        meta_style))
    s.append(Spacer(1, 2*cm))
    s.append(kv_table([
        ("Objective",
         "Verify whether normal operating data forms a single cluster in "
         "feature/latent space before applying Deep SVDD."),
        ("Methods",
         "A. FFT features + PCA/t-SNE   |   B. GMM BIC/AIC   |   "
         "C. Pairwise distance KDE   |   D. Orbit image pixel space"),
        ("Key Finding",
         "Normal data exhibits a multi-modal structure, not a single cluster. "
         "Per-RPM independent Deep SVDD training is required."),
    ], col_widths=[3.5*cm, CONTENT_W - 3.5*cm]))
    s.append(PageBreak())

    # ── 1. Dataset Overview ───────────────────────────────────────────────────
    s.append(Paragraph("1.  Dataset Overview", h1_style))
    s.append(section_line())
    s.append(Paragraph(
        "The analysis targets real measured normal BIN files collected from "
        "the RCP (Reactor Coolant Pump) Vibration Monitoring System (VMS) "
        "of a nuclear power plant. Files cover two operating conditions "
        "(3600 rpm and 1200 rpm). Each file contains 24 channels sampled at "
        "40 kHz for 10 seconds (400,000 samples per channel, float32 LE, "
        "non-interleaved).",
        body_style))
    s.append(data_table(
        ["Series", "Files", "RPM", "1X Freq.", "Path"],
        [
            ["250-series", "28", "3600 rpm", "60 Hz", "data/raw/normal_3600rpm/"],
            ["251-series", "82", "1200 rpm", "20 Hz", "data/raw/normal_1200rpm/"],
            ["Total",     "110", "—",        "—",     "data/raw/normal/"],
        ],
        col_widths=[2.8*cm, 1.8*cm, 2.5*cm, 2.8*cm, CONTENT_W - 9.9*cm]
    ))
    s.append(Spacer(1, 0.3*cm))
    s.append(Paragraph(
        "Four bearing XY channel pairs are used: (ch0/1), (ch4/5), (ch10/11), "
        "(ch16/17). The FFT feature vector consists of 1X, 2X, 3X harmonic "
        "amplitudes and RMS per channel, yielding a 32-dimensional vector per file.",
        body_style))
    s.append(PageBreak())

    # ── 2. Method A ───────────────────────────────────────────────────────────
    s.append(Paragraph("2.  Method A — FFT Features + PCA / t-SNE", h1_style))
    s.append(section_line())

    s.append(Paragraph("2.1  Methodology", h2_style))
    s.append(Paragraph(
        "FFT was applied to eight bearing channels to extract 1X, 2X, and 3X "
        "harmonic amplitudes and RMS (1X = 60 Hz for 3600 rpm; 1X = 20 Hz for "
        "1200 rpm; search bandwidth ±2 Hz). Feature vectors were standardised "
        "(StandardScaler) and reduced to 2D via PCA. A second pipeline applied "
        "PCA (10 components) followed by t-SNE (2D) to visualise cluster "
        "structure.",
        body_style))

    s.append(Paragraph("2.2  Results", h2_style))
    s.append(fig("pca_tsne.png", width=CONTENT_W, height=CONTENT_W * 0.42))
    s.append(Paragraph(
        "Figure 1.  PCA 2D scatter (left) and t-SNE 2D scatter (right). "
        "Blue = 1200 rpm, Red = 3600 rpm.",
        caption_style))
    s.append(fig("pca_components.png",
                 width=CONTENT_W * 0.7, height=CONTENT_W * 0.38))
    s.append(Paragraph(
        "Figure 2.  PCA explained variance ratio (bars) and cumulative "
        "variance (line).",
        caption_style))

    s.append(Paragraph("2.3  Key Metrics", h2_style))
    s.append(data_table(
        ["Metric", "Value", "Interpretation"],
        [
            ["PC1 explained variance", "72.4%",
             "1X/RMS amplitude difference (RPM) dominates"],
            ["PC2 explained variance", "12.8%",
             "Secondary variation (vibration directionality)"],
            ["PCs needed for 90% variance", "3",
             "32-dim data effectively 3-dimensional"],
        ],
        col_widths=[5.5*cm, 3*cm, CONTENT_W - 8.5*cm]
    ))
    s.append(Spacer(1, 0.3*cm))
    s.append(conclusion_box(
        "Conclusion A: RPM conditions are completely separated. "
        "A sub-cluster exists within 1200 rpm (top-right vs. bottom of t-SNE). "
        "Single-hypersphere assumption is violated."
    ))
    s.append(PageBreak())

    # ── 3. Method B ───────────────────────────────────────────────────────────
    s.append(Paragraph(
        "3.  Method B — GMM BIC / AIC Component Selection", h1_style))
    s.append(section_line())

    s.append(Paragraph("3.1  Methodology", h2_style))
    s.append(Paragraph(
        "Gaussian Mixture Models (GMM) were fitted for k = 1 to 6 components "
        "on the 32-dimensional standardised features from Method A. The "
        "Bayesian Information Criterion (BIC) and Akaike Information Criterion "
        "(AIC) were used to select the optimal number of clusters. Three "
        "subsets were analysed: (1) All files (n=110), "
        "(2) 1200 rpm only (n=82), (3) 3600 rpm only (n=28).",
        body_style))

    s.append(Paragraph("3.2  Results", h2_style))
    s.append(fig("gmm_bic_aic.png", width=CONTENT_W, height=CONTENT_W * 0.35))
    s.append(Paragraph(
        "Figure 3.  Normalised GMM BIC and AIC scores per group "
        "(minimum = optimal k).",
        caption_style))
    s.append(fig("gmm_ellipses.png", width=CONTENT_W, height=CONTENT_W * 0.38))
    s.append(Paragraph(
        "Figure 4.  GMM ellipses (2-sigma) overlaid on PCA 2D space "
        "for optimal k (BIC).",
        caption_style))

    s.append(Paragraph("3.3  Optimal Component Summary", h2_style))
    s.append(data_table(
        ["Subset", "n", "BIC k*", "AIC k*", "Remarks"],
        [
            ["All", "110", "3", "4",
             "1200 rpm main + sub-cluster + 3600 rpm"],
            ["1200 rpm", "82", "1", "6",
             "BIC=1 but wide ellipse; BIC/AIC disagree"],
            ["3600 rpm", "28", "2", "3",
             "Main cluster + outliers; small n, low confidence"],
        ],
        col_widths=[2.5*cm, 1.5*cm, 2*cm, 2*cm, CONTENT_W - 8*cm]
    ))
    s.append(Spacer(1, 0.3*cm))
    s.append(conclusion_box(
        "Conclusion B: Overall k=3 confirms that a single SVDD sphere "
        "would have an excessively large radius. "
        "1200 rpm is a borderline case (BIC=1 but high variance; AIC disagrees). "
        "3600 rpm judgement deferred due to insufficient samples."
    ))
    s.append(PageBreak())

    # ── 4. Method C ───────────────────────────────────────────────────────────
    s.append(Paragraph(
        "4.  Method C — Pairwise Distance Distribution + KDE", h1_style))
    s.append(section_line())

    s.append(Paragraph("4.1  Methodology", h2_style))
    s.append(Paragraph(
        "Without any cluster assumption, the full pairwise Euclidean distance "
        "matrix was computed and Kernel Density Estimation (KDE) was applied "
        "to visualise the distribution shape directly. Three pair types were "
        "compared: intra-1200 rpm, intra-3600 rpm, and inter-RPM. "
        "Nearest-Neighbour (NN) distance distributions and a distance matrix "
        "heatmap were produced as additional evidence. "
        "Hartigan's Dip Test was used to quantify unimodality.",
        body_style))

    s.append(Paragraph("4.2  Distance Statistics", h2_style))
    s.append(data_table(
        ["Pair Type", "Pairs", "Mean Dist.", "Std Dev.", "inter/intra Ratio"],
        [
            ["intra-1200 rpm", "6,642", "3.157", "1.621", "—"],
            ["intra-3600 rpm",   "756", "5.859", "3.755", "—"],
            ["inter-RPM",      "2,296", "11.787", "1.877",
             "vs 1200: 3.73x   /   vs 3600: 2.01x"],
        ],
        col_widths=[3.5*cm, 2*cm, 2.8*cm, 2.5*cm, CONTENT_W - 10.8*cm]
    ))

    s.append(Paragraph("4.3  Visualisations", h2_style))
    s.append(fig("pairwise_kde.png", width=CONTENT_W, height=CONTENT_W * 0.35))
    s.append(Paragraph(
        "Figure 5.  Pairwise distance KDE.  "
        "(Left) All intra vs. inter-RPM;  "
        "(Centre) 1200 rpm only;  (Right) 3600 rpm only.",
        caption_style))

    nn_w = CONTENT_W * 0.54
    hm_w = CONTENT_W * 0.44
    s.append(Table(
        [[Image(str(IMG_DIR / "nn_distance.png"),
                width=nn_w, height=nn_w * 0.42),
          Image(str(IMG_DIR / "pairwise_heatmap.png"),
                width=hm_w, height=hm_w * 1.05)]],
        colWidths=[nn_w, hm_w]
    ))
    s.append(Paragraph(
        "Figure 6.  (Left) Nearest-Neighbour distance distributions by RPM.  "
        "(Right) Pairwise distance matrix heatmap (sorted by RPM; "
        "brighter = smaller distance).",
        caption_style))

    s.append(Paragraph("4.4  Hartigan Dip Test", h2_style))
    s.append(data_table(
        ["Subset", "Dip Statistic", "p-value", "Verdict", "Note"],
        [
            ["intra-1200 rpm", "0.0002", "1.000", "unimodal",
             "n^2 pairs dilute signal; low power"],
            ["intra-3600 rpm", "0.0013", "1.000", "unimodal", "Same issue"],
            ["inter-RPM",      "0.0004", "1.000", "unimodal", "Same issue"],
        ],
        col_widths=[3.5*cm, 3*cm, 2*cm, 2.5*cm, CONTENT_W - 11*cm]
    ))
    s.append(Paragraph(
        "Note: Dip Test loses power when applied to n^2 pairwise distances. "
        "KDE visual inspection (Figure 5) is the more reliable evidence.",
        note_style))
    s.append(Spacer(1, 0.3*cm))
    s.append(conclusion_box(
        "Conclusion C: inter/intra distance ratio of 3.73x numerically "
        "confirms RPM separation. "
        "KDE curves for both 1200 rpm and 3600 rpm show bimodal shapes, "
        "indicating sub-clusters within each RPM condition."
    ))
    s.append(PageBreak())

    # ── 5. Method D ───────────────────────────────────────────────────────────
    s.append(Paragraph(
        "5.  Method D — Orbit Image Pixel-Space Analysis", h1_style))
    s.append(section_line())

    s.append(Paragraph("5.1  Methodology", h2_style))
    s.append(Paragraph(
        "To validate cluster structure in the same input space used by "
        "Deep SVDD, orbit images were generated on-the-fly from BIN files "
        "(adaptive axis_lim, img_size = 64 for PCA; img_size = 128 for "
        "gallery/mean comparison). Three analyses were performed: "
        "(1) per-RPM sample gallery (visual inspection), "
        "(2) per-RPM mean orbit image comparison, "
        "(3) pixel-space (4x64x64 = 16,384 dims) PCA + t-SNE.",
        body_style))

    s.append(Paragraph("5.2  Orbit Image Gallery", h2_style))
    s.append(Paragraph("1200 rpm", h3_style))
    s.append(fig("orbit_gallery_1200rpm.png",
                 width=CONTENT_W * 0.72, height=CONTENT_W * 0.9))
    s.append(Paragraph(
        "Figure 7.  Orbit images for 4 sample 1200 rpm files "
        "(rows) x 4 bearing pairs (columns). "
        "Consistent elliptical / circular orbit patterns are observed "
        "across files.",
        caption_style))

    s.append(Paragraph("3600 rpm", h3_style))
    s.append(fig("orbit_gallery_3600rpm.png",
                 width=CONTENT_W * 0.72, height=CONTENT_W * 0.9))
    s.append(Paragraph(
        "Figure 8.  Orbit images for 4 sample 3600 rpm files. "
        "Shapes differ markedly across files: grid pattern (BKG_2507), "
        "vertical line (BKG_2506131), horizontal line (BKG_2506101).",
        caption_style))

    s.append(PageBreak())

    s.append(Paragraph(
        "5.3  Mean Orbit Comparison (1200 rpm vs. 3600 rpm)", h2_style))
    s.append(fig("orbit_mean_compare.png",
                 width=CONTENT_W, height=CONTENT_W * 0.48))
    s.append(Paragraph(
        "Figure 9.  Mean orbit images per RPM for all four bearing pairs. "
        "1200 rpm shows clear elliptical patterns (Bearings 2-4); "
        "3600 rpm shows near-zero amplitude or abnormal shapes throughout.",
        caption_style))

    s.append(Paragraph("5.4  Pixel-Space PCA + t-SNE", h2_style))
    s.append(fig("orbit_pca_tsne.png", width=CONTENT_W, height=CONTENT_W * 0.42))
    s.append(Paragraph(
        "Figure 10.  PCA 2D (left) and t-SNE 2D (right) of orbit pixel vectors. "
        "1200 rpm splits into 2 tight clusters; "
        "3600 rpm fragments into 4-5 scattered micro-clusters.",
        caption_style))

    s.append(data_table(
        ["Metric", "Value", "Note"],
        [
            ["Pixel feature dimension", "16,384", "4 channels x 64x64"],
            ["PC1 explained variance",  "46.4%",
             "Lower than FFT features (72.4%) — high-dim pixel structure"],
            ["PC2 explained variance",  "15.5%", ""],
            ["t-SNE clusters (1200 rpm)", "2",
             "Two highly compact, well-separated clusters"],
            ["t-SNE clusters (3600 rpm)", "4-5",
             "Fragmented micro-clusters — severe heterogeneity"],
        ],
        col_widths=[5*cm, 3.5*cm, CONTENT_W - 8.5*cm]
    ))

    s.append(Spacer(1, 0.4*cm))
    s.append(warn_box(
        "WARNING — 3600 rpm Data Quality Issue Detected\n"
        "All 28 files are classified into three anomalous groups:\n"
        "  Group A (3 files, 2506_10-12): ch1 AC-RMS ~ 0.002 V — near-zero amplitude\n"
        "  Group B (15 files, 2506_13-25): ch1 fixed at 10.44 V (range = 0.000) "
        "— suspected sensor disconnection / saturation\n"
        "  Group C (10 files, 2506_26 to 2507_08): both channels AC-RMS < 0.001 V "
        "— noise floor; ADC quantisation grid artefact visible in orbit images"
    ))
    s.append(Spacer(1, 0.3*cm))
    s.append(conclusion_box(
        "Conclusion D: Multi-modal structure confirmed in orbit pixel space. "
        "The 3600 rpm normal dataset has a fundamental data quality problem. "
        "Data cleansing / re-collection is mandatory before Deep SVDD training."
    ))
    s.append(PageBreak())

    # ── 6. Summary & Recommendations ─────────────────────────────────────────
    s.append(Paragraph(
        "6.  Summary and Deep SVDD Recommendations", h1_style))
    s.append(section_line())

    s.append(Paragraph("6.1  Cross-Method Summary", h2_style))
    s.append(data_table(
        ["Method", "Key Finding", "Implication for Deep SVDD"],
        [
            ["A  FFT + PCA/t-SNE",
             "Complete RPM separation; 1200 rpm sub-cluster",
             "Single SVDD infeasible"],
            ["B  GMM BIC",
             "Overall k=3; 1200 rpm high variance; 3600 rpm k=2",
             "Per-RPM split required"],
            ["C  Distance KDE",
             "inter/intra ratio 3.73x; bimodal KDE in each RPM",
             "Per-RPM split required"],
            ["D  Orbit pixel",
             "1200 rpm 2 clusters; 3600 rpm 4-5 micro-clusters; data faults",
             "3600 rpm data cleansing critical"],
        ],
        col_widths=[3*cm, 7*cm, CONTENT_W - 10*cm]
    ))

    s.append(Spacer(1, 0.4*cm))
    s.append(Paragraph("6.2  3600 rpm File Classification", h2_style))
    s.append(data_table(
        ["Group", "Date Range", "Files", "ch1 Characteristic",
         "Suspected Cause", "Fit for Training"],
        [
            ["A", "2506_10-12",      "3",
             "AC-RMS = 0.002 V", "Near-zero ch1 amplitude",  "No"],
            ["B", "2506_13-25",     "15",
             "range = 0.000 V; DC = 10.44 V",
             "Sensor disconnection / saturation", "No"],
            ["C", "2506_26 to 2507_08", "10",
             "AC-RMS < 0.001 V", "Noise floor (ADC quantisation)", "No"],
        ],
        col_widths=[1.5*cm, 3.8*cm, 1.5*cm, 4*cm, 4*cm,
                    CONTENT_W - 14.8*cm]
    ))

    s.append(Spacer(1, 0.4*cm))
    s.append(Paragraph("6.3  Recommendations", h2_style))
    for title, body in [
        ("(1)  Train per-RPM independent Deep SVDD models",
         "Train a separate Deep SVDD for 1200 rpm and 3600 rpm respectively. "
         "This minimises the hypersphere radius and improves anomaly detection "
         "sensitivity for each operating condition."),
        ("(2)  Cleanse / supplement 3600 rpm data",
         "None of the 28 existing 3600 rpm files produce valid orbit images. "
         "Remove files with saturated/disconnected sensors and collect "
         "additional normal 3600 rpm measurements before training."),
        ("(3)  Investigate 1200 rpm sub-cluster origin",
         "The two sub-clusters observed in t-SNE and KDE (Method A, C, D) "
         "should be traced back to measurement date, operating load, or "
         "sensor configuration differences via file metadata."),
        ("(4)  Consider Soft-boundary SVDD or outlier fraction",
         "If outliers cannot be fully removed from training data, use "
         "Soft-boundary SVDD loss or set a non-zero nu (outlier fraction) "
         "to prevent the hypersphere from collapsing around outliers."),
    ]:
        s.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", ParagraphStyle("rec",
                fontSize=10, fontName="Helvetica-Bold",
                textColor=TEAL, spaceBefore=10, spaceAfter=2)),
            Paragraph(body, body_style),
        ]))

    s.append(Spacer(1, 0.6*cm))
    s.append(HRFlowable(width="100%", thickness=1.5, color=NAVY, spaceAfter=8))
    s.append(conclusion_box(
        "Final Conclusion: Normal operating data does NOT form a single cluster "
        "and violates the single-hypersphere assumption of Deep SVDD. "
        "Per-RPM independent training is required. "
        "3600 rpm data must be cleansed and supplemented before any model "
        "training can proceed."
    ))

    return s


# ── Build PDF ─────────────────────────────────────────────────────────────────
def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
        title="RCP VMS Normal Data Cluster Analysis Report",
        author="rcpvms-gen",
    )
    doc.build(build_story(),
              onFirstPage=add_page_number,
              onLaterPages=add_page_number)
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"PDF generated: {OUT_PDF.resolve()}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
