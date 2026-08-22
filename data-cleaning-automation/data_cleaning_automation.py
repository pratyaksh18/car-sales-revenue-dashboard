"""
data_cleaning_automation.py
================================================================
A REUSABLE Data Cleaning & Reporting Automation tool.

Point it at ANY raw CSV and it will automatically:
  1. Profile data quality issues (missing values, duplicates,
     inconsistent text formatting, inconsistent dates, invalid
     numeric entries)
  2. Clean the data (standardize text, parse dates, impute missing
     values, remove duplicates, fix invalid numbers)
  3. Log every single action taken, in order, as an audit trail
  4. Generate a visual before/after data-quality report (PNG charts)
  5. Export the cleaned dataset + a written summary report

Usage:
    from data_cleaning_automation import DataCleaningAutomation

    pipeline = DataCleaningAutomation("raw_file.csv")
    pipeline.run()
    pipeline.save_report("output_folder/")

This is intentionally generic — column types are auto-detected, so
the same tool works on any similarly-shaped transactional dataset
(sales, inventory, HR records, etc.) without code changes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 130


class DataCleaningAutomation:
    """Automated data cleaning + reporting pipeline for tabular CSV data."""

    # Column names that look like they contain dates (auto-detected too,
    # but an explicit hint list makes behaviour predictable)
    DATE_HINTS = ["date", "dob", "timestamp"]

    def __init__(self, filepath: str, id_column: str = None):
        self.filepath = filepath
        self.id_column = id_column
        self.raw_df = None
        self.df = None
        self.log = []              # audit trail of every action taken
        self.quality_before = {}
        self.quality_after = {}
        self.date_columns = []
        self.numeric_columns = []
        self.categorical_columns = []

    # ------------------------------------------------------------
    def _record(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        print(f"  - {message}")

    # ------------------------------------------------------------
    def load(self):
        print(f"\n[1/6] Loading raw data from '{self.filepath}' ...")
        self.raw_df = pd.read_csv(self.filepath)
        self.df = self.raw_df.copy()
        self._record(f"Loaded {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        return self

    # ------------------------------------------------------------
    def _detect_column_types(self):
        for col in self.df.columns:
            lower = col.lower()
            if any(hint in lower for hint in self.DATE_HINTS):
                self.date_columns.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)

    # ------------------------------------------------------------
    def profile(self, stage="before"):
        """Compute a data-quality snapshot (missing %, duplicates, etc.)."""
        n_rows = len(self.df)
        missing = self.df.isnull().sum()
        missing_pct = (missing.sum() / (n_rows * len(self.df.columns))) * 100
        duplicates = self.df.duplicated().sum()

        inconsistent_text = 0
        for col in self.categorical_columns:
            vals = self.df[col].dropna().astype(str)
            has_whitespace = (vals != vals.str.strip()).sum()
            normalized_dupes = vals.str.strip().str.lower().duplicated(keep=False).sum() - \
                                vals.duplicated(keep=False).sum()
            inconsistent_text += has_whitespace + max(normalized_dupes, 0)

        snapshot = {
            "rows": n_rows,
            "missing_cells": int(missing.sum()),
            "missing_pct": round(missing_pct, 2),
            "duplicate_rows": int(duplicates),
            "inconsistent_text_entries": int(inconsistent_text),
            "missing_by_column": missing[missing > 0].to_dict(),
        }
        if stage == "before":
            self.quality_before = snapshot
        else:
            self.quality_after = snapshot
        return snapshot

    # ------------------------------------------------------------
    def clean_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        if removed:
            self._record(f"Removed {removed} exact duplicate row(s)")
        return self

    # ------------------------------------------------------------
    def standardize_text(self):
        """Trim whitespace and normalize casing on categorical columns
        (title-case for consistency), while keeping IDs untouched."""
        for col in self.categorical_columns:
            if col == self.id_column:
                continue
            original = self.df[col].copy()
            self.df[col] = self.df[col].astype(str).str.strip()
            # Only title-case columns that look like categories (not free text/IDs)
            avg_len = self.df[col].dropna().astype(str).str.len().mean()
            if avg_len is not None and avg_len < 25:
                def _smart_title(v):
                    # Preserve short all-caps acronyms (e.g. "CNG") instead of
                    # mangling them into "Cng" via naive title-casing
                    v = str(v)
                    if v.isupper() and len(v) <= 4:
                        return v
                    return v.title()
                self.df[col] = self.df[col].apply(_smart_title)
            changed = (original.astype(str).str.strip() != self.df[col]).sum()
            if changed:
                self._record(f"Standardized text formatting in '{col}' ({changed} values affected)")
        return self

    # ------------------------------------------------------------
    def standardize_dates(self):
        for col in self.date_columns:
            before_na = self.df[col].isnull().sum()
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=False,
                                           format='mixed')
            after_na = self.df[col].isnull().sum()
            unparsed = after_na - before_na
            self._record(f"Parsed '{col}' into a consistent datetime format"
                          + (f" ({unparsed} unparseable values set to NaT)" if unparsed > 0 else ""))
        return self

    # ------------------------------------------------------------
    def fix_invalid_numeric(self, rules: dict = None):
        """rules: {column: 'positive'} marks values <= 0 as invalid -> NaN,
        which then get imputed in impute_missing()."""
        rules = rules or {}
        for col, rule in rules.items():
            if col not in self.df.columns:
                continue
            if rule == 'positive':
                mask = self.df[col] <= 0
                n = mask.sum()
                if n:
                    self.df.loc[mask, col] = np.nan
                    self._record(f"Flagged {n} invalid non-positive value(s) in '{col}' for re-imputation")
            elif rule == 'non_negative':
                mask = self.df[col] < 0
                n = mask.sum()
                if n:
                    self.df.loc[mask, col] = np.nan
                    self._record(f"Flagged {n} invalid negative value(s) in '{col}' for re-imputation")
        return self

    # ------------------------------------------------------------
    def impute_missing(self):
        for col in self.numeric_columns:
            n_missing = self.df[col].isnull().sum()
            if n_missing:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                self._record(f"Imputed {n_missing} missing value(s) in '{col}' with median ({median_val:.1f})")

        for col in self.categorical_columns:
            n_missing = self.df[col].isnull().sum() + (self.df[col].astype(str) == 'nan').sum()
            if n_missing:
                mode_val = self.df[col].mode(dropna=True)
                mode_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
                self.df[col] = self.df[col].replace('nan', np.nan)
                self.df[col] = self.df[col].fillna(mode_val)
                self._record(f"Imputed {n_missing} missing value(s) in '{col}' with mode ('{mode_val}')")
        return self

    # ------------------------------------------------------------
    def recompute_dependent_columns(self, formula_cols: dict = None):
        """Optionally recompute a derived column (e.g. Revenue = Units * Price
        * (1 - Discount/100)) after cleaning, to fix any inconsistencies that
        arose from imputed inputs. formula_cols: {'Revenue (INR)': lambda df: ...}"""
        formula_cols = formula_cols or {}
        for col, fn in formula_cols.items():
            if col in self.df.columns:
                recomputed = fn(self.df)
                diff = (self.df[col] - recomputed).abs()
                mismatches = (diff > 1).sum()
                self.df[col] = recomputed.round(0)
                if mismatches:
                    self._record(f"Recomputed '{col}' from source fields to ensure consistency "
                                  f"({mismatches} rows corrected)")
        return self

    # ------------------------------------------------------------
    def run(self, numeric_validity_rules: dict = None, formula_cols: dict = None):
        self.load()
        self._detect_column_types()

        print("\n[2/6] Profiling raw data quality ...")
        self.profile(stage="before")

        print("\n[3/6] Cleaning: duplicates, text, dates ...")
        self.clean_duplicates()
        self.standardize_text()
        self.standardize_dates()
        # Re-check for duplicates that only became identical AFTER text
        # standardization (e.g. "PETROL" vs "Petrol" vs " petrol ")
        self.clean_duplicates()

        print("\n[4/6] Fixing invalid numeric entries ...")
        self.fix_invalid_numeric(numeric_validity_rules)

        print("\n[5/6] Imputing missing values ...")
        self.impute_missing()

        if formula_cols:
            self.recompute_dependent_columns(formula_cols)

        print("\n[6/6] Profiling cleaned data quality ...")
        self.profile(stage="after")

        return self

    # ------------------------------------------------------------
    def data_quality_score(self, snapshot):
        """A simple 0-100 composite score: 100 = perfect quality."""
        penalty = snapshot["missing_pct"] * 2
        penalty += (snapshot["duplicate_rows"] / max(snapshot["rows"], 1)) * 100
        penalty += (snapshot["inconsistent_text_entries"] / max(snapshot["rows"], 1)) * 20
        return max(0, round(100 - penalty, 1))

    # ------------------------------------------------------------
    def save_report(self, output_dir="."):
        import os
        os.makedirs(output_dir, exist_ok=True)

        clean_path = os.path.join(output_dir, "cleaned_data.csv")
        self.df.to_csv(clean_path, index=False)

        score_before = self.data_quality_score(self.quality_before)
        score_after = self.data_quality_score(self.quality_after)

        # --- Chart 1: Before/After quality metrics ---
        metrics = ["missing_cells", "duplicate_rows", "inconsistent_text_entries"]
        labels = ["Missing Cells", "Duplicate Rows", "Inconsistent Text"]
        before_vals = [self.quality_before[m] for m in metrics]
        after_vals = [self.quality_after[m] for m in metrics]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(x - width/2, before_vals, width, label='Before Cleaning', color='#C44E52')
        ax.bar(x + width/2, after_vals, width, label='After Cleaning', color='#55A868')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        ax.set_title("Data Quality Issues: Before vs After Cleaning")
        ax.legend()
        for i, v in enumerate(before_vals):
            ax.text(i - width/2, v + max(before_vals)*0.02, str(v), ha='center', fontsize=9)
        for i, v in enumerate(after_vals):
            ax.text(i + width/2, v + max(before_vals)*0.02, str(v), ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "chart_quality_before_after.png"), bbox_inches='tight')
        plt.close()

        # --- Chart 2: Data quality score gauge-style bar ---
        fig, ax = plt.subplots(figsize=(6, 3.2))
        bars = ax.barh(["Before Cleaning", "After Cleaning"], [score_before, score_after],
                        color=['#C44E52', '#55A868'])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Data Quality Score (0-100)")
        ax.set_title("Overall Data Quality Score")
        for bar, val in zip(bars, [score_before, score_after]):
            ax.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val}", va='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "chart_quality_score.png"), bbox_inches='tight')
        plt.close()

        # --- Chart 3: Missing values by column (before) ---
        if self.quality_before["missing_by_column"]:
            miss_df = pd.Series(self.quality_before["missing_by_column"]).sort_values(ascending=True)
            plt.figure(figsize=(6.5, 4))
            miss_df.plot(kind='barh', color='#DD8452')
            plt.title("Missing Values by Column (Raw Data)")
            plt.xlabel("Missing Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "chart_missing_by_column.png"), bbox_inches='tight')
            plt.close()

        # --- Save audit log ---
        log_path = os.path.join(output_dir, "cleaning_audit_log.txt")
        with open(log_path, "w") as f:
            f.write("DATA CLEANING AUTOMATION — AUDIT LOG\n")
            f.write("=" * 50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")
            f.write(f"\nData Quality Score: {score_before} -> {score_after} (out of 100)\n")

        # --- Save quality summary CSV ---
        summary = pd.DataFrame({
            "Metric": ["Rows", "Missing Cells", "Missing %", "Duplicate Rows",
                       "Inconsistent Text Entries", "Data Quality Score"],
            "Before": [self.quality_before["rows"], self.quality_before["missing_cells"],
                       self.quality_before["missing_pct"], self.quality_before["duplicate_rows"],
                       self.quality_before["inconsistent_text_entries"], score_before],
            "After": [self.quality_after["rows"], self.quality_after["missing_cells"],
                      self.quality_after["missing_pct"], self.quality_after["duplicate_rows"],
                      self.quality_after["inconsistent_text_entries"], score_after],
        })
        summary.to_csv(os.path.join(output_dir, "quality_summary.csv"), index=False)

        print(f"\nReport saved to '{output_dir}/':")
        print(f"  - cleaned_data.csv ({self.df.shape[0]} rows)")
        print(f"  - cleaning_audit_log.txt ({len(self.log)} actions logged)")
        print(f"  - quality_summary.csv")
        print(f"  - 3 chart PNGs")
        print(f"\nData Quality Score improved: {score_before} -> {score_after} / 100")

        return summary


# ====================================================================
# DEMO: Run the automation pipeline on the messy Car Sales dataset
# ====================================================================
if __name__ == "__main__":
    pipeline = DataCleaningAutomation("Car_Sales_RAW_messy.csv", id_column="Sale ID")

    pipeline.run(
        numeric_validity_rules={
            "Units Sold": "positive",
            "Discount (%)": "non_negative",
            "Unit Price (INR)": "positive",
        },
        formula_cols={
            "Revenue (INR)": lambda df: df["Units Sold"] * df["Unit Price (INR)"] *
                                         (1 - df["Discount (%)"] / 100)
        }
    )

    pipeline.save_report(output_dir=".")
