# astrasens
 Determination of the contrast curve and plots for the high-spatial resolution images obtained with the AstraLux instrument at Calar Alto observatory (Spain).

## Usage
 Please ask J. Lillo-Box if you want to use this code for your science.

 The code assumes that you have the following structure of folders within the `root_path` variable:

 ``root_path/11_REDUCED/YYMMDD``

for AstraLux images obtained on the night YYMMDD. The code will then create the following additional folders:
 ```
 root_path/22_ANALYSIS/Sensitivity
 root_path/22_ANALYSIS/DetectedSources
 root_path/22_ANALYSIS/Summary_plots
 ```

The general way to run `astrasens` is as follows:

```python astrasens_run.py [file] [path_to_file] [YYMMDD] ```

For example, for the image included in the example you can do:

```python astrasens_run.py  TDRIZZLE_0100_TOI5377_SDSSz__240122.fits /full_path/astrasens/11_REDUCED/ 240122 ```

Alternatively, if you want to run astrasens over a full list of files, create a list in plain ascii with 
one image per row ('example.lis') and run:

```python astrasens_run.py  /full_path_to_list/example.lis```

Or, if you want to run astrasens for all images within a given night:

```python astrasens_run.py all /full_path/astrasens/11_REDUCED/ 240122 ```



## Examples
Locating companions:

![alt text](https://github.com/jlillo/astrasens/blob/master/images/TOI5377_SDSSz__240122_0100__Residuals.png)

Determining the sensitivity curve:

![alt text](https://github.com/jlillo/astrasens/blob/master/images/TOI5377_SDSSz__240122_0100__Summary.png)

Performing aperture photometry on detected companions:

![alt text](https://github.com/jlillo/astrasens/blob/master/images/TOI-1169_SDSSz__191029_0100__AperturePhot.png)

