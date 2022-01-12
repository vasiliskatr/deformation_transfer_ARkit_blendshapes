# deformation_transfer_ARkit_blendshapes
Implementation of the deformation transfer paper (https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf) and its application in generating all the ARkit facial blend shapes for any 3D face.

We used the cool Wrap3D tool (https://www.russian3dscanner.com) to perform NRICP and deform the target neutral face to fit to the source neutral face. 

The meshes used in the data folder are only for demonstration purposes and originate from open source projects (target face mesh taken from https://github.com/ICT-VGL/ICT-FaceKit, source ARkit blend shapes and meshes taken from http://blog.kiteandlightning.la/iphone-x-facial-capture-apple-blendshapes/).


![alt text](https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes/blob/main/images/dt_flowchart.png?raw=true)


## Dependencies
* numpy
* scipy
* numba
* pandas
* plotly
