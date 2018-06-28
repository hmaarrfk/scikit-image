#!/usr/bin/env bash
set -ex

# Now configure Matplotlib to use Qt5
if [[ "${QT}" == "PySide2" ]]; then
    pip install--retries 3 -q $PIP_FLAGS pyside2
    MPL_QT_API=PySide2
    export QT_API=pyside2
else
    pip install --retries 3 -q $PIP_FLAGS pyqt5
    MPL_QT_API=PyQt5
    export QT_API=pyqt5
fi

# Is this correct for PySide2?
echo 'backend: Qt5Agg' > $MPL_DIR/matplotlibrc
echo 'backend.qt5 : '$MPL_QT_API >> $MPL_DIR/matplotlibrc

set +ex
