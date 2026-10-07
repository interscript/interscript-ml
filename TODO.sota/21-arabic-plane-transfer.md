# WO21 — Arabic plane register transfer (post-r8)

Once WO17's dual-surface verdict names the winning silver mixture
(r8a self / r8b qcri / r8c both), retrain ara-diac-plane-large with
that mixture (same plane recipe) so the ON-DEVICE family inherits the
register closure — not just the dedicated line. Gate: plane-large OOD
(18.69/11.35) moves AND in-domain stays ≤3.0.
