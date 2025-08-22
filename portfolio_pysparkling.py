#!/usr/bin/env python3
"""
Portfolio-ready Python script demonstrating data ingestion, processing,
and visualization with PySpark and Pandas.
"""

import os
import json
import csv
import datetime
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pysparkling # python3 -m pip install pysparkling
import matplotlib.pyplot as plt # python3 -m pip install matplotlib
import seaborn as sns # python3 -m pip install seaborn

# ======================
# Logging configuration
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================
# Data Reading Functions
# ======================
def read_json_file(sc: pysparkling.Context, fname: str) -> "pysparkling.RDD":
    """Read a JSON file into an RDD."""
    with open(fname, "r") as f:
        data = json.load(f)
    rdd = sc.parallelize([data])
    return rdd


def read_csv_file(sc: pysparkling.Context, fname: str) -> "pysparkling.RDD":
    """Read a CSV file into an RDD."""
    return sc.textFile(fname).mapPartitions(csv.reader)


def read_file(
    fname: Optional[str] = None, repartition: bool = False, show_log: bool = False
) -> Dict:
    """
    Read a JSON or CSV file into a PySpark RDD, optionally repartitioning.

    Args:
        fname: Filename to read. Defaults to 'sample_data.json'.
        repartition: Whether to repartition RDD based on file size.
        show_log: Whether to log messages.

    Returns:
        Dictionary containing RDD, Spark context, filename, and partition count.
    """
    if fname is None:
        # Default to JSON or CSV
        if os.path.exists("Sample_Data/sample_products.csv"):
            fname = "Sample_Data/sample_products.csv"
        elif os.path.exists("Sample_Data/sample_data.json"):
            fname = "Sample_Data/sample_data.json"
        else:
            raise FileNotFoundError("No default sample file found.")

    if not os.path.exists(fname):
        raise FileNotFoundError(f"File '{fname}' does not exist.")

    sc = pysparkling.Context()
    ext = os.path.splitext(fname)[1].lower()

    if ext == ".json":
        if show_log:
            logger.info(f"Reading JSON file: {fname}")
        rdd = read_json_file(sc, fname)
    elif ext == ".csv":
        if show_log:
            logger.info(f"Reading CSV file: {fname}")
        rdd = read_csv_file(sc, fname)
    else:
        if show_log:
            logger.warning(f"Unknown file type, reading as text: {fname}")
        rdd = sc.textFile(fname)

    num_parts = rdd.getNumPartitions()
    if repartition:
        fsize_mb = max(2, int(os.path.getsize(fname) / (1024 * 1024)))
        if num_parts != fsize_mb:
            rdd = rdd.repartition(fsize_mb)
            if show_log:
                logger.info(f"Repartitioned RDD to {fsize_mb} partitions")

    if show_log:
        logger.info(f"Number of partitions: {rdd.getNumPartitions()}")

    return {"rdd": rdd, "sc": sc, "fname": fname, "partitions": rdd.getNumPartitions()}


# ======================
# Data Extraction
# ======================
def extract_prices(df: pd.DataFrame) -> pd.Series:
    """Extract nested product prices from a JSON dataframe."""
    prices = [product["price"] for product in df["products"].iloc[0]]
    prices_series = pd.Series(prices).replace("", 0).fillna(0).astype(float)
    return prices_series.sort_values()


def extract_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Extract and clean a numeric column from a CSV dataframe."""
    series = df[column_name].replace("", 0).fillna(0).astype(float)
    return series.sort_values()


# ======================
# Visualization
# ======================
def plot_statistics(series: pd.Series, title: str = "Data Distribution") -> None:
    """Plot histogram and log-transformed line plot."""
    sns.set(style="whitegrid")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(series, bins=20, kde=True, ax=ax1)
    ax1.set_title(f"{title} - Histogram")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")
    fig1.tight_layout()
    plt.show(block=False)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(np.log10(1 + series.values), "ko-", markersize=4)
    ax2.set_title(f"{title} - Log-transformed values")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("log10(1+Value)")
    fig2.tight_layout()
    plt.show(block=True)


# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    start_time = datetime.datetime.utcnow()
    logger.info("<> START <>")

    # Read file
    result = read_file(fname=None, repartition=True, show_log=True)
    rdd = result["rdd"]
    fname = result["fname"]

    # Collect first partition
    first_partition = rdd.mapPartitionsWithIndex(lambda idx, it: it if idx == 0 else [])
    rows = list(first_partition.collect())

    # Convert to DataFrame
    ext = os.path.splitext(fname)[1].lower()
    if ext == ".json":
        df = pd.DataFrame(rows)
        series = extract_prices(df)
    elif ext == ".csv":
        header, *data = rows
        df = pd.DataFrame(data, columns=header)
        series = extract_column(df, "price")
    else:
        raise ValueError("Unsupported file format for extraction")

    logger.info(
        f"Series stats: len={len(series)}, min={np.nanmin(series)}, "
        f"mean={np.nanmean(series)}, max={np.nanmax(series)}"
    )

    # Plot
    plot_statistics(series, title=f"Statistics for {fname}")

    end_time = datetime.datetime.utcnow()
    logger.info(f"<> END [runtime={end_time - start_time}] <>")
