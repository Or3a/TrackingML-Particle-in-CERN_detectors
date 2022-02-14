# TrackingML-Particle

Language: Python 

What is happening? To explore what our universe is made of, scientists at CERN are colliding protons,
essentially recreating mini big bangs, and meticulously observing these collisions with intricate silicon detectors.
While orchestrating the collisions and observations is already a massive scientific accomplishment, analyzing the enormous amounts of data produced from the experiments 
is becoming an overwhelming challenge.

Every particle leaves a track behind, like a car leaving tire marks. 
The particle is not caught in action. Now we want to link every track to one hit that the particle created.
In every event, a large number of particles are released. They move along a path leaving behind their tracks.
They eventually hit a particle detector "surface". In this Notebook I am going to visualize these tracks in the best way possible and will also go through some more details.
I have reference from other submissions on Kaggle.

> https://www.kaggle.com/c/trackml-particle-identification/data
> Download only the detectors and the train_sample zip. Exctract the zip files and it should be all set.
> The notebook, the detectors file and the train_sample folder should be in the same folder


### Trackml library to help dealing with data
> import trackml # pip install --user git+https://github.com/LAL/trackml-library.git


