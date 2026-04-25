"""
agent/nodes/generate_report.py — LLM Report Generator + PDF Builder

This is the final node in the agent pipeline. It:

  1. Calls Groq (Llama3) with the full analysis data via the prompt
     template from agent/prompts.py to generate a written narrative.

  2. Parses the LLM response into four sections:
       - Executive Summary
       - Technical Analysis
       - Fundamental Analysis
       - Risk Assessment

  3. Renders everything into a clean, professional PDF using reportlab:
       - Header with company name, ticker, and report date
       - Key metrics table (price, RSI, volatility, P/E, sentiment)
       - Risk flags table (if any flags were raised)
       - Four narrative sections from the LLM
       - Footer with data sources and generation timestamp

  4. Saves the PDF to data/reports/{TICKER}_{DATE}.pdf

The PDF path and LLM token usage are returned in the state dict
so MLflow can log them as artifacts and metrics in Week 3.
"""

import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from groq import Groq
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, HRFlowable
)

from config import config
from agent.prompts import build_report_prompt

logger = logging.getLogger(__name__)


def generate_report(state: dict) -> dict:
    """
    Generate the LLM narrative and compile the final PDF report.

    Args:
        state: Fully populated state dict (after all 5 prior nodes).

    Returns:
        state augmented with:
            report_path         str   Absolute path to the saved PDF
            report_filename     str   Just the filename, e.g. "AAPL_2026-04-25.pdf"
            llm_narrative       str   Raw text returned by Groq
            llm_tokens_used     int   Total tokens consumed (prompt + completion)
            llm_latency_seconds float Time taken for the Groq call
            report_status       str   "success" | "error"
    """
    ticker = state.get("ticker", "UNKNOWN")
    company = state.get("company_name", ticker)
    logger.info("Generating report for %s", ticker)

    # ── Step 1: Call Groq LLM ─────────────────────────────────────────────────
    try:
        prompt = build_report_prompt(state)
        narrative, tokens_used, llm_latency = _call_groq(prompt)
    except Exception as exc:
        logger.error("LLM call failed for %s: %s", ticker, exc)
        return {
            **state,
            "report_status": "error",
            "report_error": f"LLM call failed: {exc}",
        }

    # ── Step 2: Parse LLM response into sections ──────────────────────────────
    sections = _parse_sections(narrative)

    # ── Step 3: Build PDF ─────────────────────────────────────────────────────
    config.ensure_dirs()
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{ticker}_{date_str}.pdf"
    report_path = config.REPORTS_DIR / filename

    try:
        _build_pdf(
            path=report_path,
            ticker=ticker,
            company=company,
            state=state,
            sections=sections,
            date_str=date_str,
        )
        logger.info("PDF report saved: %s", report_path)
    except Exception as exc:
        logger.error("PDF generation failed for %s: %s", ticker, exc, exc_info=True)
        return {
            **state,
            "llm_narrative": narrative,
            "llm_tokens_used": tokens_used,
            "llm_latency_seconds": llm_latency,
            "report_status": "error",
            "report_error": f"PDF generation failed: {exc}",
        }

    return {
        **state,
        "report_path": str(report_path),
        "report_filename": filename,
        "llm_narrative": narrative,
        "llm_tokens_used": tokens_used,
        "llm_latency_seconds": round(llm_latency, 2),
        "report_sections": sections,
        "report_status": "success",
    }


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> tuple[str, int, float]:
    """
    Send the prompt to Groq and return (response_text, total_tokens, latency_seconds).
    """
    client = Groq(api_key=config.GROQ_API_KEY)

    start = time.time()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior financial analyst. Write clear, factual, "
                    "structured financial research reports based strictly on the "
                    "data provided. Never fabricate numbers or make up statistics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    latency = time.time() - start

    text = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0

    logger.info("Groq call complete: %d tokens, %.2fs", tokens, latency)
    return text, tokens, latency


# ── Section parser ────────────────────────────────────────────────────────────

def _parse_sections(narrative: str) -> dict:
    """
    Parse the LLM response into four named sections.

    Expected format (instructed in the prompt):
        ## EXECUTIVE SUMMARY
        ...text...
        ## TECHNICAL ANALYSIS
        ...text...
        ## FUNDAMENTAL ANALYSIS
        ...text...
        ## RISK ASSESSMENT
        ...text...
    """
    section_keys = [
        "EXECUTIVE SUMMARY",
        "TECHNICAL ANALYSIS",
        "FUNDAMENTAL ANALYSIS",
        "RISK ASSESSMENT",
    ]

    sections: dict[str, str] = {k: "" for k in section_keys}

    # Split on ## headers
    parts = re.split(r"##\s+", narrative)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for key in section_keys:
            if part.upper().startswith(key):
                content = part[len(key):].strip()
                sections[key] = content
                break

    # Fallback: if parsing fails, dump everything into Executive Summary
    if all(v == "" for v in sections.values()):
        sections["EXECUTIVE SUMMARY"] = narrative.strip()
        logger.warning("Section parsing failed — dumping full narrative into Executive Summary")

    return sections


# ── PDF builder ───────────────────────────────────────────────────────────────

def _build_pdf(
    path: Path,
    ticker: str,
    company: str,
    state: dict,
    sections: dict,
    date_str: str,
) -> None:
    """
    Build and save the PDF report using reportlab Platypus (flow-based layout).
    """
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = _build_styles()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("FinSight Financial Research", styles["header_sub"]))
    story.append(Paragraph(f"{company} ({ticker})", styles["header_main"]))
    story.append(Paragraph(f"Report Generated: {date_str}", styles["header_sub"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.2 * inch))

    # ── Risk badge ────────────────────────────────────────────────────────────
    risk_level = state.get("risk_level", "unknown").upper()
    risk_color = {
        "LOW": colors.HexColor("#2ecc71"),
        "MEDIUM": colors.HexColor("#f39c12"),
        "HIGH": colors.HexColor("#e74c3c"),
        "CRITICAL": colors.HexColor("#8e44ad"),
    }.get(risk_level, colors.grey)

    risk_style = ParagraphStyle(
        "risk_badge",
        parent=styles["normal"],
        textColor=colors.white,
        backColor=risk_color,
        fontSize=11,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=6,
        borderPadding=(4, 8, 4, 8),
    )
    anomaly_text = "⚠ ANOMALY DETECTED" if state.get("anomaly_detected") else "✓ NO ANOMALIES"
    story.append(Paragraph(f"RISK LEVEL: {risk_level}   |   {anomaly_text}", risk_style))
    story.append(Spacer(1, 0.15 * inch))

    # ── Key metrics table ─────────────────────────────────────────────────────
    story.append(Paragraph("Key Metrics", styles["section_heading"]))
    story.append(Spacer(1, 0.05 * inch))

    def fmt(v, prefix="$", decimals=2, default="N/A"):
        if v is None:
            return default
        if isinstance(v, float):
            return f"{prefix}{v:.{decimals}f}" if prefix else f"{v:.{decimals}f}"
        return f"{prefix}{v}" if prefix else str(v)

    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Current Price", fmt(state.get("current_price")),
         "52W High", fmt(state.get("week_52_high"))],
        ["RSI (14-day)", f"{state.get('rsi_14', 'N/A')} ({state.get('rsi_signal', '')})",
         "52W Low", fmt(state.get("week_52_low"))],
        ["Volatility (30d)", f"{state.get('volatility_30d_pct', 'N/A')}% ({state.get('volatility_signal', '')})",
         "MA30", fmt(state.get("ma_30"))],
        ["Momentum (10d)", f"{state.get('momentum_10d_pct', 'N/A')}%",
         "MA200", fmt(state.get("ma_200"))],
        ["P/E Ratio", fmt(state.get("pe_ratio"), prefix=""),
         "EPS", fmt(state.get("eps"))],
        ["Profit Margin", f"{(state.get('profit_margin') or 0) * 100:.1f}%",
         "Beta", fmt(state.get("beta"), prefix="")],
        ["News Sentiment", f"{state.get('sentiment_label', 'N/A')} ({state.get('average_sentiment_score', 'N/A')})",
         "Analyst Target", fmt(state.get("analyst_target_price"))],
    ]

    col_widths = [1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch]
    metrics_table = Table(metrics_data, colWidths=col_widths)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── Flags table (if any) ──────────────────────────────────────────────────
    flags = state.get("flags", [])
    if flags:
        story.append(Paragraph("Risk Flags", styles["section_heading"]))
        story.append(Spacer(1, 0.05 * inch))
        flag_data = [["Severity", "Flag", "Description"]]
        for f in flags:
            flag_data.append([
                f["severity"].upper(),
                f["name"].replace("_", " ").title(),
                Paragraph(f["description"][:120] + ("..." if len(f["description"]) > 120 else ""),
                          styles["small"]),
            ])
        flag_table = Table(flag_data, colWidths=[0.8 * inch, 1.5 * inch, 4.9 * inch])
        severity_colors = {"HIGH": colors.HexColor("#e74c3c"), "MEDIUM": colors.HexColor("#f39c12"), "LOW": colors.HexColor("#2ecc71")}
        flag_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]
        for i, f in enumerate(flags, start=1):
            bg = severity_colors.get(f["severity"].upper(), colors.white)
            flag_style.append(("BACKGROUND", (0, i), (0, i), bg))
            flag_style.append(("TEXTCOLOR", (0, i), (0, i), colors.white))
            flag_style.append(("FONTNAME", (0, i), (0, i), "Helvetica-Bold"))
        flag_table.setStyle(TableStyle(flag_style))
        story.append(flag_table)
        story.append(Spacer(1, 0.2 * inch))

    # ── Narrative sections ────────────────────────────────────────────────────
    section_display_names = {
        "EXECUTIVE SUMMARY": "Executive Summary",
        "TECHNICAL ANALYSIS": "Technical Analysis",
        "FUNDAMENTAL ANALYSIS": "Fundamental Analysis",
        "RISK ASSESSMENT": "Risk Assessment",
    }

    for key, display_name in section_display_names.items():
        text = sections.get(key, "").strip()
        if not text:
            continue
        story.append(Paragraph(display_name, styles["section_heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dee2e6")))
        story.append(Spacer(1, 0.05 * inch))
        # Split on paragraph breaks
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace("\n", " "), styles["body"]))
                story.append(Spacer(1, 0.08 * inch))
        story.append(Spacer(1, 0.1 * inch))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
    story.append(Spacer(1, 0.1 * inch))
    footer_text = (
        f"<b>Data Sources:</b> Yahoo Finance (yfinance), Alpha Vantage, NewsAPI &nbsp;|&nbsp; "
        f"<b>LLM:</b> Groq / {config.LLM_MODEL} &nbsp;|&nbsp; "
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; "
        f"<i>For educational purposes only. Not financial advice.</i>"
    )
    story.append(Paragraph(footer_text, styles["footer"]))

    doc.build(story)


# ── Style definitions ─────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "header_main": ParagraphStyle(
            "header_main",
            parent=base["Title"],
            fontSize=22,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "header_sub": ParagraphStyle(
            "header_sub",
            parent=base["Normal"],
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#6c757d"),
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base["Heading2"],
            fontSize=13,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a1a2e"),
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#212529"),
            leading=15,
            alignment=TA_JUSTIFY,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontSize=8,
            fontName="Helvetica",
            textColor=colors.HexColor("#495057"),
            leading=11,
        ),
        "normal": base["Normal"],
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontSize=7,
            fontName="Helvetica",
            textColor=colors.HexColor("#6c757d"),
            alignment=TA_CENTER,
        ),
    }


# ── Quick manual test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from agent.nodes.fetch_price import fetch_price_data
    from agent.nodes.fetch_fundamentals import fetch_fundamentals
    from agent.nodes.fetch_news import fetch_news
    from agent.nodes.analyze import analyze
    from agent.nodes.detect_anomaly import detect_anomaly

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"Running full pipeline for {ticker}...")
    state = fetch_price_data(ticker)
    state.update(fetch_fundamentals(ticker))
    news = fetch_news(state.get("company_name", ticker), ticker)
    state.update(news)
    state = analyze(state)
    state = detect_anomaly(state)
    result = generate_report(state)

    print(f"\nReport status : {result.get('report_status')}")
    print(f"PDF saved to  : {result.get('report_path')}")
    print(f"LLM tokens    : {result.get('llm_tokens_used')}")
    print(f"LLM latency   : {result.get('llm_latency_seconds')}s")
