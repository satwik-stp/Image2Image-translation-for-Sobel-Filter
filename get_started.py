import os

"""
This code creates directories and subdirectories used in this take home assignment
"""

if not os.path.exists("data"):
    # The directory for storing the data
    os.mkdir("data")
    # Sub directory of "data" -> "raw" for storing original, immutable data dump
    os.mkdir("data/raw")
    # Sub directory of "data" -> "interim" for storing transformed data
    os.mkdir("data/interim")
    # Sub directory of "data" -> "processed" for storing final canonical dataset for modeling
    os.mkdir("data/processed")

if not os.path.exists("figures"):
    # The directory for storing the graphics and figures generated for reporting
    os.mkdir("figures")

if not os.path.exists("models"):
    # The directory for storing trained and serialized models, model predictions and summaries
    os.mkdir("models")

if not os.path.exists("reports"):
    # The directory for storing generated analysis as PDF, LaTex etc
    os.mkdir("reports")