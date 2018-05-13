#!/usr/bin/env bash
set -ex

PY=$TRAVIS_PYTHON_VERSION

# Matplotlib settings - do not show figures during doc examples
if [[ $MINIMUM_REQUIREMENTS == 1 || $TRAVIS_OS_NAME == "osx" ]]; then
    MPL_DIR=$HOME/.matplotlib
else
    MPL_DIR=$HOME/.config/matplotlib
fi

mkdir -p $MPL_DIR
touch $MPL_DIR/matplotlibrc

if [[ $TRAVIS_OS_NAME == "osx" ]]; then
    echo 'backend : Template' > $MPL_DIR/matplotlibrc
fi

section "Flake8.test"
flake8 --exit-zero --exclude=test_*,six.py skimage doc/examples viewer_examples
section_end "Flake8.test"


section "Install.optional.dependencies"

# Install most of the optional packages
if [[ $OPTIONAL_DEPS == 1 ]]; then
    pip install --retries 3 -q -r ./requirements/optional.txt $WHEELHOUSE
fi

# Install Qt and then update the Matplotlib settings
if [[ $QT == "PyQt4" ]]; then
    # http://stackoverflow.com/a/9716100
    LIBS=( PyQt4 sip.so )

    VAR=( $(which -a python$PY) )

    GET_PYTHON_LIB_CMD="from distutils.sysconfig import get_python_lib; print (get_python_lib())"
    LIB_VIRTUALENV_PATH=$(python -c "$GET_PYTHON_LIB_CMD")
    LIB_SYSTEM_PATH=$(${VAR[-1]} -c "$GET_PYTHON_LIB_CMD")

    for LIB in ${LIBS[@]}
    do
        ln -sf $LIB_SYSTEM_PATH/$LIB $LIB_VIRTUALENV_PATH/$LIB
    done

elif [ "$QT" == "PySide" ] && [ -e ~/venv/bin/pyside_postinstall.py ]; then
    python ~/venv/bin/pyside_postinstall.py -install
fi

if [[ $WITH_PYAMG == 1 ]]; then
    pip install --retries 3 -q pyamg
fi

section_end "Install.optional.dependencies"


section "Build.docs.or.run.examples"

if [[ $BUILD_DOCS == 1 ]]; then
    # This fails on Travis OSX
    pip install --retries 3 -q -r ./requirements/docs.txt
    export SPHINXCACHE=$HOME/.cache/sphinx; make html
else
    # Why is this optional?
    echo 'backend : Template' > $MPL_DIR/matplotlibrc

    for f in doc/examples/*/*.py; do
        python "$f"
        if [ $? -ne 0 ]; then
            exit 1
        fi
    done
fi

section_end "Build.docs.or.run.examples"


section "Run.doc.applications"

for f in doc/examples/xx_applications/*.py; do
    python "$f"
    if [ $? -ne 0 ]; then
        exit 1
    fi
done

section_end "Run.doc.applications"


section "Test.with.optional.dependencies"

# Now configure Matplotlib to use Qt4
if [[ $QT == "PyQt4" ]]; then
    MPL_QT_API=PyQt4
    export QT_API=pyqt
elif [[ $QT == "PyQt4" ]]; then
        MPL_QT_API=PyQt5
        export QT_API=pyqt5
elif [[ $QT == "PySide" ]]; then
    MPL_QT_API=PySide
    export QT_API=pyside
elif [[ $QT == "PySide2" ]]; then
    MPL_QT_API=PySide2
    export QT_API=pyside2
fi
if [[ $QT == "PyQt4" || $QT == "PySide" ]]; then
    echo 'backend: Qt4Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt4 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
elif [[ $QT == "PyQt5" || $QT == "PySide2" ]]; then
    # Is this correct for PySide2?
    echo 'backend: Qt5Agg' > $MPL_DIR/matplotlibrc
    echo 'backend.qt5 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc
fi

# run tests again with optional dependencies to get more coverage
if [[ $OPTIONAL_DEPS == 1 ]]; then
    TEST_ARGS="$TEST_ARGS --cov=skimage"
fi

# Show what's installed
pip list
pytest $TEST_ARGS skimage

section_end "Test.with.optional.dependencies"

section "Prepare.release"
doc/release/contribs.py HEAD~10
section_end "Prepare.release"
