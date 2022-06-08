# vertical-federated-learning


### How to start

#### Set up PyVertical
Clone the PyVertical project from https://github.com/OpenMined/PyVertical. Install it following the instruction specified in the README file of the PyVertical repo.
Specify the path where you cloned PyVertical in a `.env` file, and place that file in the `vertical-federated-learning` folder.

#### Use PyVertical
Before launching any script, enter the virtual environment created while installing PyVertical:

`conda activate pyvertical-dev`

Then inside the scripts, import `sys` and our `utils` packages. Obtain the path to PyVertical and add it to the Python path with the following commands:

```
PYVERTICAL_LOCATION = utils.load_from_dotenv("PYVERTICAL_LOCATION")
sys.path.append(PYVERTICAL_LOCATION)
```



