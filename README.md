# Sebocyte Exosome Proteomics

This repository contains the Python workflow used for batch correction and downstream proteomics analysis of sebocytes treated with mesenchymal stem cell (MSC)-derived exosomes.

## Overview

The workflow was developed for label-free proteomics data acquired across two LC-MS/MS platforms. The analysis was designed to compare control sebocytes and sebocytes treated with MSC-derived exosomes after correction of platform-associated batch effects.

The workflow includes:

1. Import of Perseus-merged LFQ intensity data
2. Definition of experimental condition and LC-MS/MS platform batch variables
3. Visualization of sample-wise LFQ intensity distributions before batch correction
4. Batch-effect correction using the ComBat algorithm implemented in `neuroCombat`
5. Restoration of the original missing-value structure after batch correction
6. Filtering of proteins based on valid values across experimental groups
7. Welch's t-test for differential protein abundance analysis
8. Benjamini-Hochberg false discovery rate (FDR) correction
9. Volcano plot statistics and visualization
10. Violin plots before and after batch-effect correction

## Input data

The workflow expects an Excel file containing a Perseus-merged LFQ intensity matrix.

The default expected input path is:

```text
data/Merged_Exosome_Data.xlsx
