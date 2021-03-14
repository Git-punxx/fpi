# Intrinsic imaging analysis software

<img src="/home/remi/TDS/Programming/InProgress/Intrinsic/screenshot_numbered.png" style="zoom:50%;" />

## Analysis

To analyze an experiment, navigate to the folder using the *file explorer* **#1** and press the *Analysis* button **#5**. 

## Visualization

To visualize an already analyzed experiment, use the  *file explorer* **#1** to find the relevant *datastore* file (Same folder as the experiment, using a similar name with the *h5* suffix). Select it.

The *anatomy panel* **#2** shows a raw image from the experiment, allowing to look at the blood vessel pattern for example, useful to localize the region of interest.

The *signal panel* **#3**, can display the processed fluorescence signal in various ways which you can select using the drop-down menu **#7**:

- Average stack: Fluorescence signal computed on the average of every frame. Given all the trials each frame is averaged (all the frame #1 together, then all the #2 ...). Then the baseline average (first 30 frames, before stimulus) is subtracted from this average stack which is then divided by this same baseline, yielding the standard $\frac{\Delta F}{F}$. 
- Normalized stack: The normalized stack tries to compensate for some photo bleaching occurring during the acquisition. The time course of the average (over space, all pixels) $\frac{\Delta F}{F}$ is fitted with an exponential function: $decay(t) = A*e^{k*t}$. This exponential decay is then subtracted from the time course of every pixel. This yields flatter fluorescence profiles.
- Max projection: default value. Shows the maximum value of each pixel of the normalized stack. Very useful to detect the response as a white spot.

When average or normalized stack are displayed, one can scroll through the stack using the slider **#9** that also shows the current frame number on its right.

The contrast of the fluorescence image can be tweaked using the slider **#6** by adjusting which fluorescence value is displayed as white. This is for display only, it does not affect the underlying data.

The *signal panel* **#3** has a yellow ROI **#3'** which can be moved and stretched. The value of the pixels selected by this box are averaged and displayed in the *time course panel* **#4**. By default the normalized stack is used but if the average stack is shown, then those values are used. Using this ROI, one can confirm that a white spot is indeed a plausible physiological response, because its time course follows the stimulation. The ROI in the *anatomy panel* **#2** is linked to this one, though it can not be moved independently. This way, one can easily mark, on the anatomy image the localization of the response detected on the signal image.

The *comment field* **#8** is to be completed by the experimenter. These metadata are saved in the same *datastore* file ensuring that one always know where the data is coming from. It also makes automated analysis much easier later on. It is a good idea to indicate : genotype, optional treatment, stimulus used, and any relevant information. To fill it, just type and then press `Enter` to validate and save this comment.

All of the analyzed data, the comments as well as some information on the raw data are saved in the *datastore.h5* file. This is standard format, for which a lot of tools exist. Opening them in Python or Matlab is made possible by the use of standard techniques. Quickly exploring them is also possible using softwares like *HDFview* or *Panoply* for example.

## Exports

The four buttons at the bottom of the window allows to create some extra files:

+ **10** *Movie*: If the required Python packages are installed, allows the export of a movie of the current signal
+ **11** *Save response*: Save a png image of the fluorescence signal. It tries its best to extract pixels that are responsive to the stimulation. To do so it correlates every pixel fluorescence time course to a theoretical responsive time course. If the correlation is significant then the pixel is included in the response, if not it is excluded. It usually provides a good first map of the response.
+ **12** *Save time course*: Export the fluorescence time course visible in the *time course panel* **#4** to an easily reusable SVG file.
+ **13** *Excel export*: Export various analyzed data to Excel-compatible CSV files.
  + One *`response`* file contains the average fluorescence over time of the significantly responsive area (the same one that is exported as an image using the *Save response* button #**11**). It also saves the peak fluorescence as well as the number of pixels which are significantly modulated.
  + One *`timecourse`* file contains the fluorescence time course displayed in the *time course panel* **#4**  (average and standard deviation).
  + One *`all_pixels`*` file contains the fluorescence time course of all responding pixels. Each line is a pixel, each column a time point
