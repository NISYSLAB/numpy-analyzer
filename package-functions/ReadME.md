# Python based tool for analysis and visualization of electrophysiological signals
The tool consists of functions that can be used directly in the user's code or through the GUI for easy viewing and visualization of electrophsyiological signals. 

This folder contains the viewer source code, GUI design file, GUI main code, demos of the viewer's functions, and report containing full explaination of the viewer's functionality.

### Viewer Structure
The viewer folder contains the viewer package. The GUI imports and uses the viewer package for functionality. The viewer package functions can be used directly from code by importing the viewer
```
import viewer
```
or by importing the needed module from the viewer

```
from viewer import visualization
```
or 
```
import viewer.visualization
```
For full explaination of the viewer modules and functions refer to the report [here](Viewer-Report.pdf)


### GUI 
To run the GUI download the project repo, navigate to the package-functions folder, and then run the main file.
```
python main.py
```

### Future Work
- **Generate pdf and html reports:** this feature would allow the users to experiment in the GUI then download the results they want to keep in the form of a report. This could be useful for presentations and documentation of studies.
- **Integerate SSCWT and CWT graphs into the GUI:** Currently the generated SSCWT and CWT graphs are displayed as html pages due to an error in the generated file from plolty's to_html() method. It would be nice to find a workaround and integerate them into the GUI itself.








