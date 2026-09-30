"""
ContextIQ - UI Styles & Custom CSS
Provides a modern, refined dark theme with sleek cards, badges, and clean typography.
"""

CUSTOM_CSS = """
<style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }

    /* Header styling */
    .main-header {
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }

    /* Section Cards */
    .section-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Status badge pills */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-badge.online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-badge.offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Source badge */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #0F172A;
        border: 1px solid #38BDF8;
        color: #E0F2FE;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Chunk cards in expander */
    .chunk-container {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
    }
    .chunk-meta {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
    }
    .chunk-text {
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.5;
        white-space: pre-wrap;
    }
</style>
"""
