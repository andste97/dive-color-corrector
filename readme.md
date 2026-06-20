## Dive and underwater image and video color correction

**Sample images**

![Example](./examples/example.jpg)

**Sample video**

[![Video](https://img.youtube.com/vi/NEpl41-LMBs/0.jpg)](https://www.youtube.com/watch?v=NEpl41-LMBs)


### Setup
```
$ pip install -r requirements.txt
```


### For images
```
$ python correct.py image /my/raw.png /my/corrected.png
```

### For videos
```
$ python correct.py video /my/raw.mp4 /my/corrected.mp4
```

### Running the tests
A unit test suite lives in the `tests/` directory and runs automatically on
every push via GitHub Actions.

```
$ pip install -r requirements-dev.txt
$ pytest
```

## GUI
You can either download the [desktop softwares](https://bornfree.github.io/dive-color-corrector/) or build one yourself.

![GUI](./examples/gui.jpg)

### Building the GUI
Uncomment the libraries needed for GUI in `requirements.txt` and re-run `pip install`.

MacOS (via Py2App)
```
$ py2applet --make-setup dcc.py
$ python setup.py py2app
```

Windows (via PyInstaller)
```
$ python -m PyInstaller -n "Dive Color Corrector" -F -w -i .\logo\logo.ico dcc.py
```

Linux (via PyInstaller)
```
$ pyinstaller -n "Dive Color Corrector" -F -w -i ./logo/logo.png dcc.py
```

Final builds will be available in 'dist' folder

### Automated release builds
Whenever a new version tag (e.g. `v1.0.0`) is pushed, the
[`Release`](./.github/workflows/release.yml) GitHub Actions workflow bundles the
app for Windows, macOS and Linux with PyInstaller and attaches the executables
to a GitHub Release.

```
$ git tag v1.0.0
$ git push origin v1.0.0
```



### Share
If this repo was useful, please considering [sharing the word](https://twitter.com/intent/tweet?url=https://github.com/bornfree/dive-color-correction&text=Correct%20your%20dive%20footage%20with%20Python%20#scuba%20#gopro%20#python%20#opencv) on Twitter.

### Inspiration
This repo was inspired by the algorithm at https://github.com/nikolajbech/underwater-image-color-correction.
