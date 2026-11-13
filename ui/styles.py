"""
SmartResume AI - Premium Light-Theme Design System
Inspired by Linear, Notion, Stripe, and Vercel.
Palette:
- Primary Background: #FFFFFF
- Secondary Background: #F8FAFC
- Accent: Royal Blue #2563EB, Violet #7C3AED, Teal #0D9488
- Text: Charcoal #0F172A, Slate #64748B
- Border: #E2E8F0
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0F172A;
}

/* App Background */
.stApp {
    background-color: #F8FAFC;
}

/* Header & Navbar Styling */
header[data-testid="stHeader"] {
    background-color: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #E2E8F0;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 6px;
}
.sidebar-brand-mark {
    align-items: center;
    background: #2563EB;
    border-radius: 10px;
    color: #FFFFFF;
    display: flex;
    font-size: 0.95rem;
    font-weight: 700;
    height: 34px;
    justify-content: center;
    width: 34px;
}
.sidebar-brand strong { color: #172033; display: block; font-size: 1rem; }
.sidebar-brand span { color: #64748B; display: block; font-size: 0.72rem; margin-top: 1px; }

/* Modern Card Container */
.saas-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

.saas-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
}

/* KPI Metric Cards */
.kpi-container {
    display: flex;
    flex-direction: column;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #10B981;
    margin-top: 6px;
    font-weight: 500;
}

/* Brand Badge */
.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #EFF6FF;
    color: #2563EB;
    border: 1px solid #DBEAFE;
    margin-bottom: 12px;
}

/* Status Badges */
.status-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
}
.status-success { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.status-warning { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.status-info { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.status-danger { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF;
    font-weight: 500;
    font-size: 0.9rem;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    transition: all 0.15s ease-in-out;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}

/* Secondary Button Outline */
.stButton > button[kind="secondary"] {
    background-color: #FFFFFF;
    color: #334155;
    border: 1px solid #CBD5E1;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #F8FAFC;
    border-color: #94A3B8;
}

/* Data Tables */
.dataframe {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    font-size: 0.86rem !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.9rem;
    color: #64748B;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
}
</style>
"""

import plotly.io as pio

def get_plotly_theme():
    """Returns a polished Light-Theme configuration for Plotly interactive charts."""
    return {
        "layout": {
            "font": {"family": "Inter, sans-serif", "color": "#0F172A"},
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "colorway": ["#2563EB", "#7C3AED", "#0D9488", "#F59E0B", "#EC4899", "#6366F1"],
            "margin": {"l": 40, "r": 30, "t": 40, "b": 40},
            "xaxis": {"gridcolor": "#F1F5F9", "linecolor": "#E2E8F0"},
            "yaxis": {"gridcolor": "#F1F5F9", "linecolor": "#E2E8F0"}
        }
    }

