"""Build the Power BI Dashboard Guide as a polished .docx."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Power_BI_Dashboard_Guide.docx"
FIG = ROOT / "figures"

NAVY = RGBColor(0x0B, 0x1F, 0x4D)
ACCENT = RGBColor(0x06, 0x99, 0xC9)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x6B, 0x72, 0x80)


def shade(cell, color_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement('w:shd')
    sh.set(qn('w:val'), 'clear')
    sh.set(qn('w:color'), 'auto')
    sh.set(qn('w:fill'), color_hex)
    tc_pr.append(sh)


def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Calibri"; r.font.size = Pt(22); r.font.bold = True
    r.font.color.rgb = NAVY


def add_h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Calibri"; r.font.size = Pt(15); r.font.bold = True
    r.font.color.rgb = ACCENT


def add_h3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.name = "Calibri"; r.font.size = Pt(12); r.font.bold = True
    r.font.color.rgb = INK


def add_p(doc, text, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Calibri"; r.font.size = Pt(11)
    r.italic = italic
    r.font.color.rgb = INK


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(item)
        r.font.name = "Calibri"; r.font.size = Pt(11)
        r.font.color.rgb = INK


def add_code(doc, code_text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(code_text)
    r.font.name = "Consolas"; r.font.size = Pt(9.5)
    r.font.color.rgb = RGBColor(0x0F, 0x2D, 0x52)


def add_table(doc, header, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Light Grid Accent 1"
    t.autofit = True
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        p = c.paragraphs[0]
        r = p.add_run(h)
        r.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.name = "Calibri"; r.font.size = Pt(11)
        shade(c, "0B1F4D")
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            c = t.rows[ri].cells[ci]
            c.text = ""
            r = c.paragraphs[0].add_run(str(val))
            r.font.name = "Calibri"; r.font.size = Pt(10)
            r.font.color.rgb = INK


# ---------------------------------------------------------------------------
doc = Document()

# Page margins
section = doc.sections[0]
section.top_margin = Cm(2.0); section.bottom_margin = Cm(2.0)
section.left_margin = Cm(2.0); section.right_margin = Cm(2.0)

# ---------- Cover ----------
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = title.add_run("Cloudburst Risk Prediction System")
r.font.name = "Calibri"; r.font.size = Pt(32); r.font.bold = True
r.font.color.rgb = NAVY

sub = doc.add_paragraph()
r = sub.add_run("Power BI Dashboard Build Guide  |  Uttarakhand, India")
r.font.name = "Calibri"; r.font.size = Pt(15); r.font.color.rgb = ACCENT
r.italic = True

add_p(doc,
      "This document is the step-by-step recipe for assembling the production "
      "Power BI dashboard that consumes the dataset produced by the ML pipeline. "
      "It covers data import, the data model, recommended DAX measures, page "
      "layouts, slicers, and a custom dark theme aligned with the Streamlit GUI.",
      italic=False)

# ---------- 1 - Inputs ----------
add_h1(doc, "1. Inputs Provided by the Pipeline")
add_p(doc, "After running the Python pipeline, the following file is the single "
           "source of truth for Power BI:")
add_bullets(doc, ["data/powerbi_dataset.csv  -  513 rows, 23 columns, "
                  "one row per (location x date)."])
add_p(doc, "Column reference:")
add_table(doc,
    header=["Column", "Type", "Description"],
    rows=[
        ["Date", "Date", "Observation date"],
        ["Year, Month, MonthName, Season", "Calendar", "Pre-built calendar attributes"],
        ["Location", "Text", "Site name (e.g., 'Kedarnath (Rudraprayag)')"],
        ["District", "Text", "Uttarakhand district"],
        ["Latitude, Longitude", "Decimal", "Geo coordinates for the map visual"],
        ["PRECTOT", "Decimal", "Daily rainfall, mm/day"],
        ["T2M, RH2M, WS2M", "Decimal", "Temp (degC), humidity (%), wind speed (m/s)"],
        ["Rainfall_Intensity", "Whole #", "Bucket 0-5 (trace ... extreme)"],
        ["Humidity_Rainfall", "Decimal", "Interaction feature"],
        ["Heat_Index", "Decimal", "Apparent-temperature proxy"],
        ["Cloudburst", "Whole #", "Ground-truth label (0 / 1)"],
        ["Predicted_Probability", "Decimal", "Model-output probability [0..1]"],
        ["Predicted_Cloudburst", "Whole #", "0/1 at tuned threshold"],
        ["Risk_Level", "Text", "Low / Medium / High"],
    ])

# ---------- 2 - Data import ----------
add_h1(doc, "2. Importing the Data")
add_h2(doc, "2.1 Get Data > Text/CSV")
add_bullets(doc, [
    "Open Power BI Desktop and select Home > Get data > Text/CSV.",
    "Browse to data/powerbi_dataset.csv and click Open.",
    "In the preview, click Transform Data to open Power Query.",
])
add_h2(doc, "2.2 Power Query cleanup")
add_bullets(doc, [
    "Set Date column type = Date.",
    "Set Latitude / Longitude type = Decimal Number; mark them with Data Category = Latitude / Longitude.",
    "Set PRECTOT, T2M, RH2M, WS2M, Humidity_Rainfall, Heat_Index, Predicted_Probability = Decimal Number.",
    "Set Year / Month / Cloudburst / Predicted_Cloudburst / Rainfall_Intensity = Whole Number.",
    "Right-click MonthName and select Sort By Column > Month so calendar slicers stay ordered.",
    "Click Close & Apply.",
])

# ---------- 3 - Data model ----------
add_h1(doc, "3. Data Model")
add_h2(doc, "3.1 Add a calendar table")
add_p(doc, "Create a Date dimension so time intelligence works correctly. "
           "Modeling > New table:")
add_code(doc,
"""Calendar = ADDCOLUMNS (
    CALENDAR ( DATE(1990, 1, 1), DATE(2026, 12, 31) ),
    "Year",      YEAR  ( [Date] ),
    "Month",     MONTH ( [Date] ),
    "MonthName", FORMAT ( [Date], "MMM" ),
    "Quarter",   "Q" & QUARTER ( [Date] ),
    "MonthYear", FORMAT ( [Date], "MMM yyyy" )
)""")
add_p(doc, "Mark this table as date table (Modeling > Mark as date table > Date).")

add_h2(doc, "3.2 Relationships")
add_table(doc,
    header=["From", "To", "Cardinality", "Direction"],
    rows=[
        ["Cloudburst[Date]", "Calendar[Date]", "Many-to-One", "Single"],
    ])
add_p(doc, "All other dimensions (Location, District, Risk_Level) live inside the "
           "fact table for a simple star-shape that is easy to slice.")

# ---------- 4 - DAX measures ----------
add_h1(doc, "4. DAX Measures")
add_p(doc, "Create the following measures inside a new measure table called Measures.")

measures = [
    ("Total Events",
     "Total Events = SUM ( cloudburst_dataset[Cloudburst] )"),
    ("High Risk Areas",
     "High Risk Areas = CALCULATE ( DISTINCTCOUNT ( cloudburst_dataset[Location] ),\n"
     "    cloudburst_dataset[Risk_Level] = \"High\" )"),
    ("Avg Rainfall (mm)",
     "Avg Rainfall (mm) = AVERAGE ( cloudburst_dataset[PRECTOT] )"),
    ("Max Daily Rainfall",
     "Max Daily Rainfall = MAX ( cloudburst_dataset[PRECTOT] )"),
    ("Predicted Risk Pct",
     "Predicted Risk Pct = AVERAGE ( cloudburst_dataset[Predicted_Probability] ) * 100"),
    ("Cloudburst Rate",
     "Cloudburst Rate = DIVIDE ( SUM ( cloudburst_dataset[Cloudburst] ),\n"
     "    COUNTROWS ( cloudburst_dataset ) )"),
    ("Events YoY %",
     "Events YoY % = VAR thisYear = [Total Events]\n"
     "VAR lastYear = CALCULATE ( [Total Events], DATEADD ( 'Calendar'[Date], -1, YEAR ) )\n"
     "RETURN DIVIDE ( thisYear - lastYear, lastYear )"),
    ("Monsoon Events",
     "Monsoon Events = CALCULATE ( [Total Events],\n"
     "    cloudburst_dataset[Season] = \"Monsoon\" )"),
]
for name, code in measures:
    add_h3(doc, name)
    add_code(doc, code)

# ---------- 5 - Pages and visuals ----------
add_h1(doc, "5. Report Pages & Visuals")

add_h2(doc, "5.1 Page 1 - Executive Overview")
add_p(doc, "Goal: a single glance answer to 'are we at risk?'")
add_bullets(doc, [
    "Card: Total Events",
    "Card: High Risk Areas",
    "Card: Avg Rainfall (mm)",
    "Card: Cloudburst Rate (formatted as %)",
    "Filled Map (visual: Map): Location field on Latitude / Longitude, "
    "Bubble size = [Total Events], Colour saturation = [Predicted Risk Pct].",
    "Stacked column: Year on X, [Total Events] on Y, legend = Risk_Level.",
    "Slicers (top of page): District, Year, Risk_Level.",
])

add_h2(doc, "5.2 Page 2 - Rainfall vs Cloudbursts")
add_bullets(doc, [
    "Scatter chart: PRECTOT (X-axis), RH2M (Y-axis), legend = Cloudburst, size = Predicted_Probability.",
    "Add a constant line at PRECTOT = 100 mm (Format > Analytics > Constant line).",
    "Line chart: MonthName (X), [Avg Rainfall (mm)] (line), [Total Events] (column on secondary axis).",
])

add_h2(doc, "5.3 Page 3 - District Risk Analysis")
add_bullets(doc, [
    "Matrix visual: Rows = District, Columns = MonthName, Values = [Total Events]. "
    "Apply conditional formatting > Background color > rules: 0 = #052035, >5 = #ef4444.",
    "Donut: Risk_Level vs [Total Events].",
    "Treemap: Location vs [Total Events] - quickly shows which sites dominate.",
])

add_h2(doc, "5.4 Page 4 - Feature Impact")
add_bullets(doc, [
    "Bar chart of feature importances. Use a small Python script visual (or import "
    "models/metadata.json into Power BI as a separate table) to plot Feature vs "
    "Importance. The columns are exposed in the JSON under feature_importance.",
    "Decomposition tree: Analyze [Total Events], Explain by District, then Season, then Risk_Level.",
])

# ---------- 6 - Slicers / theme ----------
add_h1(doc, "6. Slicers, Filters and Theme")
add_h2(doc, "6.1 Page-level slicers (recommended on every page)")
add_bullets(doc, [
    "District (dropdown).",
    "Year (between, range slider).",
    "Risk_Level (tile slicer with the three values).",
])

add_h2(doc, "6.2 Custom theme")
add_p(doc, "View > Themes > Browse for themes - point to powerbi_theme.json "
           "shipped alongside this guide.")
add_p(doc, "The palette is calibrated to match the Streamlit GUI:")
add_table(doc,
    header=["Token", "Hex", "Use"],
    rows=[
        ["Background", "#0B1424", "Page canvas"],
        ["Surface", "#10223F", "Card backgrounds"],
        ["Primary", "#2563EB", "Charts (positive, neutral)"],
        ["Accent", "#06B6D4", "Lines, highlights"],
        ["Warning", "#F59E0B", "Medium risk"],
        ["Danger", "#EF4444", "High risk / cloudburst"],
        ["Success", "#10B981", "Low risk"],
    ])

# ---------- 7 - Refresh strategy ----------
add_h1(doc, "7. Refresh and Operations")
add_bullets(doc, [
    "Re-run the pipeline (data_pipeline.py + train_model.py) on a schedule "
    "(daily during monsoon, weekly otherwise).",
    "In Power BI Desktop, hit Home > Refresh after the CSV updates.",
    "If you publish to the Power BI Service, configure a Personal Gateway "
    "pointing at the data/powerbi_dataset.csv path and schedule refresh.",
])

# ---------- 8 - Appendix ----------
add_h1(doc, "8. Appendix - Sample Visual Reference")
add_p(doc, "The five charts below are produced by visualizations.py and mirror the "
           "Power BI visuals - useful both as a visual specification and as a "
           "fallback for offline reports.")

for img_name, caption in [
    ("feature_importance.png",     "A. Random Forest feature importance."),
    ("rainfall_vs_cloudburst.png", "B. Rainfall vs humidity, coloured by label."),
    ("risk_distribution.png",      "C. Predicted risk distribution."),
    ("monthly_trend.png",          "D. Monthly cloudburst trend vs mean rainfall."),
    ("district_heatmap.png",       "E. Cloudburst events by district & month."),
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    img_path = FIG / img_name
    if img_path.exists():
        p.add_run().add_picture(str(img_path), width=Cm(15))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True; r.font.size = Pt(9.5); r.font.color.rgb = MUTED

doc.save(str(OUT))
print("Wrote", OUT)
