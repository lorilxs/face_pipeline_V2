"""Stage5 (SCRFD bbox) + Stage10 (InsightFace 106-landmark) temporal stability audit.

Read-only analysis tooling. Consumes Stage5/Stage10 output data and produces
overlay videos, jitter statistics (JSON) and a Markdown report. It never
modifies any model or pipeline code.
"""
