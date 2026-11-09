# Data

Four datasets are public on Mendeley Data under CC BY 4.0 and one on GitHub. They are not copied
into this repository; download each one and save it here under the file name shown.

| Save as | Dataset | Records | Source |
|---|---|---|---|
| `jamalpur_2024.csv` | Jamalpur 250-Bedded General Hospital, 2024 | 1,523 | https://doi.org/10.17632/6fsrsk3mb8.2 |
| `dhaka_2025.csv` | Dhaka hospitals, 2025 (NS1-antigen label) | 1,329 | https://doi.org/10.17632/jsbmtk8hty.4 |
| `multisite_fmkp6fwrrk.csv` | Multi-site Bangladesh | 2,085 | https://doi.org/10.17632/fmkp6fwrrk.1 |
| `kalai_xrsbyjs24t.csv` | Upazila Health Complex, Kalai, Joypurhat | 1,003 | https://doi.org/10.17632/xrsbyjs24t.1 |
| `riya_dhaka_2023.csv` | Two Dhaka hospitals, 2023 (file `CBC Report.csv`) | 320 | https://github.com/mritunjoychk17/Dengue-Prediction-in-Bangladesh-Using-Machine-Learning |

Check that you have the same file versions we used:

```bash
md5sum -c data/CHECKSUMS.md5
```

The Jamalpur and Dhaka 2025 datasets are used for model development and bidirectional external
validation, and the Dhaka 2023 dataset (Riya et al.) as a third external cohort. The multi-site and
Kalai datasets are used in the data-quality audit only (see Section 3.1 of the paper).

## Dataset citations

- M. A. S. Nirob, A. K. M. F. K. Siam, P. Bishshash, and M. Assaduzzaman, "Dengue fever hematological dataset: Clinical insights for improved diagnosis and patient management," Mendeley Data, v2, 2025, doi: 10.17632/6fsrsk3mb8.2.
- A. Thakur and S. Nasrin, "Comprehensive dengue hematology and clinical dataset from Bangladesh for machine learning and diagnostic research," Mendeley Data, v4, 2025, doi: 10.17632/jsbmtk8hty.4.
- M. S. Siam, M. R. Islam, M. Assaduzzaman, and A. Mahmud, "Comprehensive hematological profiles and dengue-associated clinical data from Bangladesh: A multivariate dataset for epidemiological and clinical research," Mendeley Data, v1, 2025, doi: 10.17632/fmkp6fwrrk.1.
- M. M. H. Mim, M. Assaduzzaman, A. Mahmud, and M. A. S. Nirob, "Predictive clinical dataset for dengue fever using vital signs and blood parameters," Mendeley Data, v1, 2024, doi: 10.17632/xrsbyjs24t.1.
- N. J. Riya, M. Chakraborty, and R. Khan, "Artificial intelligence-based early detection of dengue using CBC data," IEEE Access, vol. 12, pp. 112355–112367, 2024, doi: 10.1109/ACCESS.2024.3443299 (dataset at https://github.com/mritunjoychk17/Dengue-Prediction-in-Bangladesh-Using-Machine-Learning).
