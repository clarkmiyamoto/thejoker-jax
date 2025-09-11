# How to get data from SDSS APOGEE

1. Download the data using `download.py`
```
python download.py --top=<NUM_DOWNLOADS> --visits=<NUM_MIN_VISTS>
```
It saves the data as a `data.pkl` file.

2. Visualize / inspect the data using `rv.py`. Note only plots 16 at a time.
```
python rv.py --slice=<WHICH_16_RVs?> --filename=<PATH_TO_data.pkl>
```

# GUI
Use Jupyter Notebook at a proxy for exploring the 
data in a more interactive way. I have this in` rv_explorer.ipynb`