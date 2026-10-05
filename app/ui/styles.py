"""
ContextIQ - UI Styles & Custom CSS
Provides a modern, refined dark theme with sleek cards, badges, and clean typography.
"""

CUSTOM_CSS = """
<style>
    /* Responsive main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Professional Application Header */
    .contextiq-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1E293B;
    }
    .contextiq-title-row {
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    .contextiq-brand {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .contextiq-tagline {
        font-size: 1.05rem;
        font-weight: 500;
        color: #38BDF8;
        margin: 0;
    }
    .contextiq-subtitle {
        font-size: 0.95rem;
        font-weight: 400;
        color: #94A3B8;
        margin-top: 0.4rem;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Section Cards */
    .section-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* System Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    .status-badge.online {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .status-badge.offline {
        background-color: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .status-badge.neutral {
        background-color: rgba(148, 163, 184, 0.12);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Source Citation Badges */
    .source-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #1E293B;
        border: 1px solid #334155;
        color: #E2E8F0;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        transition: border-color 0.15s ease;
    }
    .source-pill:hover {
        border-color: #38BDF8;
    }
    .source-pill-page {
        color: #38BDF8;
        font-weight: 600;
    }

    /* Chunk Cards in Retrieval Inspector */
    .chunk-container {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    .chunk-meta {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        border-bottom: 1px solid rgba(51, 65, 85, 0.6);
        padding-bottom: 0.4rem;
    }
    .chunk-text {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.55;
        white-space: pre-wrap;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Empty state card */
    .empty-state-card {
        text-align: center;
        padding: 2.25rem 1.5rem;
        color: #94A3B8;
        background: #0F172A;
        border-radius: 8px;
        border: 1px dashed #334155;
        margin-bottom: 1.25rem;
    }
    .empty-state-title {
        color: #F8FAFC;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .empty-state-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        margin: 0;
        line-height: 1.4;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .contextiq-brand {
            font-size: 1.75rem;
        }
        .contextiq-tagline {
            font-size: 0.95rem;
        }
    }
</style>
"""
