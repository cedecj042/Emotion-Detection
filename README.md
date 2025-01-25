# Automated Emotion Detection using DistilBERT Approach while integrating Robert Plutchik's Emotion Wheel for Multi-label Emotion Classification

Just run everything, the installation is also inside the notebook. 

For testing out the model, you can just run the predicting-emotion.ipynb and the first lines of code are adjustments for the pyplutchik library in line 62 - 65.
You can just change it manually inside pyplutchik folder, find the patch.py file and change line 63:

Around line 63 t.exterior should be t.exterior.coords
    vertices = concatenate([
        concatenate([asarray(t.exterior.coords)[:, :2]] +
                    [asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])

Then it will just work fine.

My Paper:
https://docs.google.com/document/d/1oJ1hJOJD1l6-suY17ZdK6kB4xiHxqfF-XcWhQfy4glw/edit?usp=drivesdk